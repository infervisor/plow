//! Speech (TTS) packet contracts: a causal codec-token LM declared as a `tts.codec_lm.v1`
//! pipeline. The programs are the ordinary causal ones; the pipeline adds the numeric contract a
//! backend-neutral speech controller needs (prompt framing, codec frame layout, stop ids,
//! sampling defaults) so plowrt binds a family by driver, never by checkpoint knowledge.

use crate::pipeline::{causal_pipeline_section, CausalPipelineSpec};
use packet::devbuild::{Model, SectionData};
use plow_asset::packet_pipeline::{PacketPipeline, PacketPipelines, SECTION};
use plow_asset::speech_tags::{Entry, Langs, SpeechTags, Unknown};

pub const DRIVER: &str = "tts.codec_lm.v1";

/// One speech family's contract. Token ids are the checkpoint tokenizer's.
pub struct SpeechProfile {
    pub tags: Option<fn(bool) -> SpeechTags>,
    /// Prompt text between the prefix and suffix ids, with `{voice}` and `{input}`.
    pub prompt_template: &'static str,
    /// A voice is valid when this renders to one vocabulary token, unless `voices` lists them.
    pub voice_token: &'static str,
    /// Named voices (multi-token names); empty = any single-token `voice_token`. `name\trender`
    /// renders `{voice}` as `render` instead of the name.
    pub voices: &'static [&'static str],
    /// Values of the template's `{language}`, the first the default; empty = no language.
    pub languages: &'static [&'static str],
    /// Longest text segment in frames (0: up to `max_new_tokens_cap`).
    pub segment_frames: u64,
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
    /// (f32 parameters, backbone repetition penalty), 2 = Maya 2 (host cb0 controller with a
    /// presence penalty and controlled stop; `repetition_penalty` is the presence penalty).
    pub depth_sampler: u64,
    /// Rendered into the template's `{instructions}` with `{instructions}` = the request's text.
    pub instructions: Option<&'static str>,
    /// Sub-bundle of an encoder-mode packet whose `act.enc_out` rows are the whole prompt's
    /// embeddings (the `overlay` role): the prompt ids only size the prefill.
    pub text_encoder: Option<&'static str>,
    /// Voice cloning through the reference encoder packet.
    pub clone: Option<CloneLayout>,
    /// Every row is a text token plus a codec frame (`plow_asset::dual_stream`, read from
    /// [`DUAL_STREAM`] beside the checkpoint): its voices and languages replace the profile's.
    pub dual_stream: bool,
    /// Streaming-output policy: leading samples quieter than `.0` dBFS before the first audible
    /// frame are not streamed, at most `.1` ms (`stream.onset_trim_*`).
    pub onset_trim: Option<(f32, u64)>,
    /// New frames a stream's first stateful codec step waits for (`stream.first_frames`; 0:
    /// undeclared, one frame).
    pub stream_first_frames: u64,
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

fn strip_tags(_: bool) -> SpeechTags {
    SpeechTags::strip()
}

fn guided_tags() -> SpeechTags {
    let mut map = SpeechTags::strip();
    map.literal_rewrite
        .extend([("[".into(), "(".into()), ("]".into(), ")".into())]);
    map
}

fn add_tags(pipe: &mut PacketPipeline, map: SpeechTags) -> Result<(), String> {
    map.validate()?;
    pipe.parameters.insert(
        plow_asset::speech_tags::VERSION_KEY.into(),
        plow_asset::speech_tags::VERSION,
    );
    pipe.strings.insert(
        plow_asset::speech_tags::KEY.into(),
        serde_json::to_string(&map).map_err(|e| e.to_string())?,
    );
    Ok(())
}

fn native_entry(text: &str, carry: bool) -> Entry {
    Entry {
        text: Some(text.into()),
        carry,
        budget_chars: Some(24),
        ..Entry::default()
    }
}

fn instruction_styles(map: &mut SpeechTags) {
    for (name, instruction) in [
        ("whisper", "Speak in a whisper."),
        ("excited", "Speak excitedly."),
        ("happy", "Speak happily."),
        ("sad", "Speak sadly."),
        ("angry", "Speak angrily."),
        ("calm", "Speak calmly."),
        ("serious", "Speak seriously."),
        ("surprised", "Speak with surprise."),
        ("friendly", "Speak in a friendly tone."),
        ("casual", "Speak casually."),
    ] {
        map.tags.insert(
            name.into(),
            Entry {
                instruction: Some(instruction.into()),
                ..Entry::default()
            },
        );
    }
}

fn orpheus_tags(_: bool) -> SpeechTags {
    let mut map = SpeechTags::strip();
    for (name, text) in [
        ("laugh", "<laugh>"),
        ("chuckle", "<chuckle>"),
        ("sigh", "<sigh>"),
        ("cough", "<cough>"),
        ("sniff", "<sniffle>"),
        ("groan", "<groan>"),
        ("yawn", "<yawn>"),
        ("gasp", "<gasp>"),
    ] {
        let mut entry = native_entry(text, false);
        entry.langs = Langs::Only(vec!["en".into()]);
        map.tags.insert(name.into(), entry);
    }
    map.tags.insert(
        "giggle".into(),
        Entry {
            alias: Some("laugh".into()),
            ..Entry::default()
        },
    );
    map
}

fn fish_tags(_: bool) -> SpeechTags {
    let mut map = SpeechTags::strip();
    map.unknown = Unknown::Passthrough;
    for (name, text, carry) in [
        ("laugh", "[laughing]", false),
        ("chuckle", "[chuckle]", false),
        ("sigh", "[sigh]", false),
        ("inhale", "[inhale]", false),
        ("exhale", "[exhale]", false),
        ("clear_throat", "[clearing throat]", false),
        ("whisper", "[whisper]", true),
        ("excited", "[excited]", true),
        ("sad", "[sad]", true),
        ("angry", "[angry]", true),
        ("emphasis", "[emphasis]", true),
    ] {
        map.tags.insert(name.into(), native_entry(text, carry));
    }
    map
}

fn breeze_tags(_: bool) -> SpeechTags {
    let mut map = SpeechTags::strip();
    for (name, en, zh) in [
        ("laugh", "(laugh)", "[笑]"),
        ("cough", "(cough)", "[咳嗽]"),
        ("clear_throat", "(clears throat)", "[清嗓子]"),
        ("sigh", "(sigh)", "[叹气]"),
    ] {
        let mut entry = native_entry(en, false);
        let mut forms = std::collections::BTreeMap::from([("en".into(), en.into())]);
        if name != "laugh" {
            forms.insert("zh".into(), zh.into());
        }
        entry.langs = Langs::Text(forms);
        map.tags.insert(name.into(), entry);
    }
    instruction_styles(&mut map);
    for name in ["calm", "friendly"] {
        map.tags.get_mut(name).unwrap().langs = Langs::Only(vec!["en".into()]);
    }
    map
}

fn qwen_tags(instructions: bool) -> SpeechTags {
    let mut map = SpeechTags::strip();
    if instructions {
        instruction_styles(&mut map);
        map.tags.insert(
            "rate".into(),
            Entry {
                instruction: Some("Speak at a {value} rate.".into()),
                ..Entry::default()
            },
        );
    }
    map
}

/// maya-research/Veena (model card): [SOH] <spk_v> text [EOH] [SOA] [SOS] -> SNAC codes, stop on
/// END_OF_SPEECH / END_OF_AI.
pub const VEENA: SpeechProfile = SpeechProfile {
    tags: Some(strip_tags),
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
    languages: &[],
    segment_frames: 0,
    dual_stream: false,
    onset_trim: None,
    stream_first_frames: 0,
};

/// canopylabs/orpheus-3b-0.1-ft (orpheus_tts `_format_prompt`): [SOH] BOS "voice: text" [EOT]
/// [EOH] [SOA] [SOS] -> the same 7-code SNAC frames as Veena, stop on END_OF_SPEECH.
pub const ORPHEUS: SpeechProfile = SpeechProfile {
    tags: Some(orpheus_tags),
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
    languages: &["en"],
    segment_frames: 0,
    dual_stream: false,
    onset_trim: None,
    stream_first_frames: 0,
};

/// fishaudio/s2-pro (fish-speech `inference.py`, no reference): system "convert the provided text
/// to speech", user "<|speaker:0|>text", assistant "<|voice|>" -> one semantic token per 21.5 Hz
/// frame (151678 + class) plus 9 depth codebooks, stop on <|im_end|>. Sampling: T 0.7, top-p 0.7,
/// top-k 30 (RAS in the depth stage).
pub const FISH: SpeechProfile = SpeechProfile {
    tags: Some(fish_tags),
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
    languages: &[],
    segment_frames: 0,
    dual_stream: false,
    onset_trim: None,
    stream_first_frames: 0,
};

/// BreezeBlue/Breeze-TTS-2 (`breeze_infer/templates.py`): each text segment `<bos>[S0]text` (voice
/// design: `[S0]<ins_bos>instruction<ins_eos>text`) runs through the T5Gemma2 text encoder; its
/// projected rows are the backbone prompt. One cb0 class per 12.5 Hz frame (EOS = class 2048) plus
/// 15 depth codebooks. Sampling (`runtime.py`, `infer.py`): T 0.9, top-k 50, top-p 1.0,
/// repetition penalty 1.1 on cb0, max 1500 frames.
pub const BREEZE: SpeechProfile = SpeechProfile {
    tags: Some(breeze_tags),
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
    languages: &["en", "zh"],
    segment_frames: 0,
    dual_stream: false,
    onset_trim: None,
    stream_first_frames: 0,
};

/// maya-research/maya2-native (`maya3` prompt.py, engine.py): `<|lang:xx|>`, `{Name}: ` (the plain
/// voice: nothing), the text, `<|audio|>` (its embedding row carries the speaker anchor) -> one cb0
/// class per 12.5 Hz frame (2048 = stop) plus 31 depth codebooks, Mimi at 24 kHz. cb0: T 0.9,
/// top-k 64, presence penalty 0.8 over 12 frames, controlled stop (`maya2.controller`); depth T 0.7,
/// top-k 64; text in pieces of about 20 s.
pub const MAYA2: SpeechProfile = SpeechProfile {
    tags: Some(strip_tags),
    prompt_template: "<|lang:{language}|>{voice}{input}<|audio|>",
    voice_token: "{voice}",
    voices: &[
        "Ahana\t{Ahana}: ",
        "Vaidehi\t{Vaidehi}: ",
        "Malini\t{Malini}: ",
        "Charvi\t{Charvi}: ",
        "Jahnavi\t{Jahnavi}: ",
        "Anirban\t{Anirban}: ",
        "Nakul\t{Nakul}: ",
        "Vasant\t{Vasant}: ",
        "Siddhant\t{Siddhant}: ",
        "Satvik\t{Satvik}: ",
        "Abhay\t{Abhay}: ",
        "Nivedita\t{Nivedita}: ",
        "Ishwari\t{Ishwari}: ",
        "Ritvik\t{Ritvik}: ",
        "Vrinda\t{Vrinda}: ",
        "Sreekar\t{Sreekar}: ",
        "Kayal\t{Kayal}: ",
        "Shaunak\t{Shaunak}: ",
        "Akshat\t{Akshat}: ",
        "Tanushree\t{Tanushree}: ",
        "plain\t",
    ],
    prefix: &[],
    suffix: &[],
    stops: &[2048],
    sample_rate: 24000,
    frame_codes: 32,
    codebook: 2048,
    frame_samples: 1920,
    // Ids past the tokenizer (`maya2_prep.py`): a prompt row of id 194549 + c is a frame row.
    audio_token_base: 194549,
    per_char_frames: 0.9,
    max_new_tokens_cap: 1500,
    temperature: 0.9,
    top_p: 1.0,
    repetition_penalty: 0.8,
    top_k: 64,
    depth: true,
    depth_sampler: 2,
    instructions: None,
    text_encoder: None,
    // `icl` (`prompt.py`): `<|lang:A|>` transcript `<|audio|>`, the reference frames, the AEOS input
    // row (id 194548), then the text's prompt; the null speaker anchor and depth memory.
    clone: Some(CloneLayout {
        head: Some(&[]),
        mid: &[],
        tail: Some(&[194548]),
        eos_code: None,
    }),
    languages: &[
        "en", "as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te",
    ],
    segment_frames: 250,
    dual_stream: false,
    onset_trim: None,
    stream_first_frames: 0,
};

/// Qwen/Qwen3-TTS-12Hz-*-CustomVoice (`qwen_tts` `generate`, `scripts/tts/qwen3_tts_prep.py`): a
/// Qwen3 talker whose rows are text plus codec frames; cb0 class 2048 = codec EOS 2150 (the other
/// control rows suppressed), at least 2 frames; a 5-layer code predictor samples codebooks 1..15.
/// Talker and predictor: T 0.7 (the published config's 0.9 babbles on ~0.7% of Seed-TTS EN
/// prompts; recipe .md), top-k 50, top-p 1; repetition penalty 1.05 on cb0.
pub const QWEN3_TTS: SpeechProfile = SpeechProfile {
    tags: Some(qwen_tags),
    prompt_template: "{input}",
    voice_token: "{voice}",
    voices: &[],
    prefix: &[],
    suffix: &[],
    stops: &[2048],
    sample_rate: 24000,
    frame_codes: 16,
    codebook: 2048,
    frame_samples: 1920,
    audio_token_base: 0,
    // As Breeze: ~2.8 frames per Mandarin character, ~0.8 per English one.
    per_char_frames: 3.2,
    max_new_tokens_cap: 1500,
    temperature: 0.7,
    top_p: 1.0,
    repetition_penalty: 1.05,
    top_k: 50,
    depth: true,
    depth_sampler: 1,
    instructions: None,
    text_encoder: None,
    clone: None,
    languages: &[],
    segment_frames: 0,
    dual_stream: true,
    // The model opens every reply with ~70 ms below -45 dBFS (the tts-bench onset level).
    onset_trim: Some((-45.0, 500)),
    // That trim drops the first frame, so the first audible chunk needs two.
    stream_first_frames: 2,
};

/// `dual_stream.json` beside a dual-stream checkpoint (`plow_asset::dual_stream::DualStream`).
pub const DUAL_STREAM: &str = "dual_stream.json";

/// `maya2_controller.json` beside a Maya 2 checkpoint (`scripts/tts/maya2_prep.py`): the cb0
/// controller, guard constants and per-language frame / duration priors.
pub const MAYA2_CONTROLLER: &str = "maya2_controller.json";

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
    voice_cond: bool,
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
    add_tags(pipe, guided_tags())?;
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
    if voice_cond {
        pipe.strings
            .insert("voice_cond.packet".into(), crate::voice_cond::PACKET.into());
    }
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
        "maya2" => Ok(&MAYA2),
        "qwen3_tts" => Ok(&QWEN3_TTS),
        other => Err(format!(
            "unknown tts profile {other:?} (known: veena, orpheus, fish, breeze, maya2, qwen3_tts)"
        )),
    }
}

pub fn speech_pipeline_section(
    model: &mut Model,
    spec: CausalPipelineSpec<'_>,
    p: &SpeechProfile,
    vocab: u32,
    dir: &std::path::Path,
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
    let mut takes_instructions = p.instructions.is_some();
    if let Some(t) = p.instructions {
        pipe.strings.insert("prompt.instructions".into(), t.into());
    }
    if !p.languages.is_empty() {
        pipe.strings
            .insert("prompt.languages".into(), p.languages.join(" "));
    }
    if p.depth_sampler == 2 {
        let path = dir.join(MAYA2_CONTROLLER);
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str::<serde_json::Value>(&text)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        pipe.strings.insert("maya2.controller".into(), text);
    }
    if p.dual_stream {
        use plow_asset::dual_stream::{
            DualStream, SpeakerConditioning, CONDITIONED_DRIVER, CONDITIONING_KEY, KEY,
        };
        let path = dir.join(DUAL_STREAM);
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let d = DualStream::parse(&text)?;
        takes_instructions |= d.instructions_template.is_some();
        if d.frame_codes as u64 != p.frame_codes {
            return Err(format!("{}: frame_codes != the profile's", path.display()));
        }
        let config_path = dir.join("config.json");
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&config_path).map_err(|e| format!("{}: {e}", config_path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", config_path.display()))?;
        if let Some(value) = config["plow_frame_embed"].get("conditioning") {
            let conditioning = SpeakerConditioning::parse(&value.to_string())?;
            if !d.voices.is_empty()
                || !p.depth
                || p.clone.is_some()
                || p.text_encoder.is_some()
                || spec.overlay_rows == 0
                || d.text_pad >= vocab
                || p.codebook > u64::from(d.none_code)
            {
                return Err("speaker conditioning requires an external speaker, frame depth and overlay rows".into());
            }
            let table = model
                .tensors
                .iter()
                .find(|t| t.name == "model.embed_tokens.weight")
                .filter(|t| t.bytes == u64::from(vocab) * u64::from(spec.hidden) * 2)
                .ok_or("speaker conditioning requires the BF16 text embedding table")?;
            pipe.tensors.insert(
                "conditioning.text_embeddings".into(),
                plow_asset::packet_pipeline::PipelineTensor {
                    name: table.name.clone(),
                    dtype: plow_asset::packet_pipeline::PipelineDType::Bf16,
                    shape: vec![u64::from(vocab), u64::from(spec.hidden)],
                },
            );
            let width = spec.hidden as usize * 2;
            let start = d.text_pad as usize * width;
            let reader = crate::checkpoint::TensorReader::open(dir)?;
            let (dtype, row) = reader.read_range(&table.name, start..start + width)?;
            if dtype != "BF16" {
                return Err("speaker conditioning requires a BF16 text-pad row".into());
            }
            let name = "in.prompt.text_pad";
            if let Some(tensor) = model.tensors.iter().find(|t| t.name == name) {
                if tensor.init.as_ref() != Some(&row) || tensor.bytes != width as u64 {
                    return Err("speaker conditioning text-pad initializer disagrees".into());
                }
            } else {
                model.tensors.push(packet::devbuild::TensorDecl {
                    name: name.into(),
                    bytes: width as u64,
                    init: Some(row),
                });
            }
            pipe.tensors.insert(
                "conditioning.text_pad_row".into(),
                plow_asset::packet_pipeline::PipelineTensor {
                    name: name.into(),
                    dtype: plow_asset::packet_pipeline::PipelineDType::Bf16,
                    shape: vec![u64::from(spec.hidden)],
                },
            );
            pipe.parameters
                .insert("conditioning.text_pad".into(), u64::from(d.text_pad));
            pipe.strings.insert(
                CONDITIONING_KEY.into(),
                serde_json::to_string(&conditioning).map_err(|e| e.to_string())?,
            );
            pipe.driver = CONDITIONED_DRIVER.into();
        } else if d.voices.is_empty() {
            return Err(
                "dual-stream checkpoint without preset voices needs explicit speaker conditioning"
                    .into(),
            );
        }
        let voices: Vec<&str> = d.voices.keys().map(String::as_str).collect();
        pipe.strings
            .insert("prompt.voices".into(), voices.join("\n"));
        let languages: Vec<&str> = std::iter::once("auto")
            .chain(d.languages.keys().map(String::as_str))
            .collect();
        pipe.strings
            .insert("prompt.languages".into(), languages.join(" "));
        let json = serde_json::to_string(&d).map_err(|e| e.to_string())?;
        pipe.strings.insert(KEY.into(), json);
    }
    if p.dual_stream || p.depth_sampler == 2 {
        pipe.parameters
            .insert("prompt.slot_prefix_version".into(), 1);
    }
    if p.depth_sampler == 2 {
        pipe.parameters.insert("stream.delivery_version".into(), 1);
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
    if p.segment_frames > 0 {
        put("tokens.segment_frames".into(), p.segment_frames);
    }
    put(
        "sampling.temperature_f32".into(),
        u64::from(p.temperature.to_bits()),
    );
    put("sampling.top_p_f32".into(), u64::from(p.top_p.to_bits()));
    put("sampling.top_k".into(), p.top_k);
    put("frames.depth".into(), u64::from(p.depth));
    if let Some((dbfs, max_ms)) = p.onset_trim {
        put(
            "stream.onset_trim_dbfs_f32".into(),
            u64::from(dbfs.to_bits()),
        );
        put("stream.onset_trim_max_ms".into(), max_ms);
    }
    if p.stream_first_frames > 0 {
        put("stream.first_frames".into(), p.stream_first_frames);
    }
    put("frames.depth_sampler".into(), p.depth_sampler);
    put(
        "sampling.repetition_penalty_f32".into(),
        u64::from(p.repetition_penalty.to_bits()),
    );
    if let Some(build) = p.tags {
        let map = build(takes_instructions);
        map.validate_binding(takes_instructions, |_| true)?;
        add_tags(pipe, map)?;
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

#[cfg(test)]
mod speech_tests {
    use super::*;
    use crate::pipeline::CausalPipelineTensors;
    use packet::devbuild::Builder;
    use plow_asset::dual_stream::{CONDITIONED_DRIVER, CONDITIONING_KEY};

    #[test]
    fn guided_tag_maps_protect_literal_brackets() {
        let mut tensors = Builder::new(2);
        let tokens = tensors.tensor("in.ids", 512 * 4);
        let positions = tensors.tensor("in.pos", 512 * 4);
        let kv_lengths = tensors.tensor("in.kvlen", 4);
        let pos_base = tensors.tensor("pos_base", 4);
        let model = Model {
            n_cu: 2,
            target: 0,
            tensors: tensors.tensors(),
            progs: (0..2).map(|_| Builder::new(2).finish()).collect(),
            kv_row_insts: Vec::new(),
            prog_t: vec![512, 1],
            gen: Vec::new(),
        };
        let spec = CausalPipelineSpec {
            name: "speech",
            max_context: 512,
            hidden: 4,
            decode_capacity: 1,
            overlay_rows: 0,
            ordered_dispatch: false,
            tensors: CausalPipelineTensors {
                tokens,
                positions,
                kv_lengths,
                overlay: None,
                overlay_index: None,
            },
        };
        let dir = std::env::temp_dir().join(format!("plow-guided-tags-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["chatterbox", "chatterbox-mtl"] {
            if name == "chatterbox-mtl" {
                std::fs::write(
                    dir.join("text_frontend.json"),
                    serde_json::json!({"rules":"[]","languages":{"en":"[]","hi":"[]"},"default_language":"en","tables":[]}).to_string(),
                ).unwrap();
            }
            let meta: PacketPipelines = serde_json::from_slice(
                &t3_pipeline_section(&model, spec, pos_base, &[], &dir, false)
                    .unwrap()
                    .data,
            )
            .unwrap();
            let pipe = &meta.pipelines[0];
            check_tag_pipeline(name, pipe, &[], false);
            let map = SpeechTags::parse(&pipe.strings[plow_asset::speech_tags::KEY]).unwrap();
            assert_eq!(map, guided_tags());
            assert!(map.literal_rewrite.contains(&("[".into(), "(".into())));
            assert!(map.literal_rewrite.contains(&("]".into(), ")".into())));
            assert_eq!(pipe.driver, GUIDED_LM_DRIVER);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn check_tag_pipeline(
        name: &str,
        pipe: &plow_asset::packet_pipeline::PacketPipeline,
        languages: &[&str],
        instructions: bool,
    ) {
        use plow_asset::speech_tags::{KEY, VERSION, VERSION_KEY};
        assert_eq!(pipe.parameters[VERSION_KEY], VERSION);
        let map = SpeechTags::parse(&pipe.strings[KEY]).unwrap();
        assert_eq!(map.syntax.markers, plow_asset::speech_tags::Markers::Strip);
        if let Some(path) = std::env::var_os("TTS_TAG_TOKENIZERS") {
            let paths: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            let tokenizer =
                tokenizers::Tokenizer::from_file(paths[name].as_str().unwrap()).unwrap();
            map.validate_binding(instructions, |text| {
                tokenizer.encode(text, false).unwrap().len() == 1
            })
            .unwrap();
            for entry in map.tags.values() {
                let mut forms: Vec<&str> = entry.text.iter().map(String::as_str).collect();
                if let Langs::Text(extra) = &entry.langs {
                    forms.extend(extra.values().map(String::as_str));
                }
                for text in forms {
                    let encoded = tokenizer.encode(text, false).unwrap();
                    assert!(!encoded.is_empty(), "{name}: {text}");
                    assert!(
                        encoded
                            .get_ids()
                            .iter()
                            .all(|&id| (id as usize) < tokenizer.get_vocab_size(true)),
                        "{name}: {text}"
                    );
                }
            }
        } else {
            map.validate_binding(instructions, |_| true).unwrap();
        }
        if let Some(path) = std::env::var_os("TTS_TAG_MAP_OUTPUT") {
            let path = std::path::PathBuf::from(path);
            let mut output: serde_json::Map<String, serde_json::Value> = if path.exists() {
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap()
            } else {
                Default::default()
            };
            output.insert(
                name.into(),
                serde_json::json!({"pipeline":pipe,"languages":languages}),
            );
            std::fs::write(path, serde_json::to_vec_pretty(&output).unwrap()).unwrap();
        }
    }

    #[test]
    fn profile_tag_map_changes_only_versioned_metadata() {
        use plow_asset::speech_tags::{SpeechTags, KEY, VERSION, VERSION_KEY};
        let mut tensors = Builder::new(2);
        let tokens = tensors.tensor("in.ids", 512 * 4);
        let positions = tensors.tensor("in.pos", 512 * 4);
        let kv_lengths = tensors.tensor("in.kvlen", 4);
        let mut model = Model {
            n_cu: 2,
            target: 0,
            tensors: tensors.tensors(),
            progs: (0..2).map(|_| Builder::new(2).finish()).collect(),
            kv_row_insts: Vec::new(),
            prog_t: vec![512, 1],
            gen: Vec::new(),
        };
        let spec = CausalPipelineSpec {
            name: "speech",
            max_context: 512,
            hidden: 4,
            decode_capacity: 1,
            overlay_rows: 0,
            ordered_dispatch: false,
            tensors: CausalPipelineTensors {
                tokens,
                positions,
                kv_lengths,
                overlay: None,
                overlay_index: None,
            },
        };
        let mut untagged = VEENA;
        untagged.tags = None;
        let control: PacketPipelines = serde_json::from_slice(
            &speech_pipeline_section(
                &mut model,
                spec,
                &untagged,
                200000,
                std::path::Path::new(""),
            )
            .unwrap()
            .data,
        )
        .unwrap();
        let mut profile = VEENA;
        profile.tags = MAYA2.tags;
        let mut tagged: PacketPipelines = serde_json::from_slice(
            &speech_pipeline_section(&mut model, spec, &profile, 200000, std::path::Path::new(""))
                .unwrap()
                .data,
        )
        .unwrap();
        assert_eq!(
            tagged.pipelines[0].parameters.remove(VERSION_KEY),
            Some(VERSION)
        );
        let map = SpeechTags::parse(&tagged.pipelines[0].strings.remove(KEY).unwrap()).unwrap();
        assert_eq!(map, SpeechTags::strip());
        assert_eq!(tagged, control);
        let dir = std::env::temp_dir().join(format!("plow-tag-profiles-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(MAYA2_CONTROLLER), "{}").unwrap();
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        let dual = serde_json::json!({
            "text_template":"{input}","instructions_template":"{instructions}","lead":3,"drop":5,
            "text_pad":2,"text_bos":3,"text_eos":4,"none_code":3072,"frame_codes":16,
            "codec_language":[2154,2156,2157],"language_at":2,"codec_auto":[2155,2156,2157],
            "codec_pad":2148,"codec_bos":2149,"languages":{"english":2050},"voices":{"voice":{"code":3061,"dialect":null}},
            "dialect_languages":["chinese","auto"]
        });
        std::fs::write(dir.join(DUAL_STREAM), dual.to_string()).unwrap();
        for (name, profile, instructions) in [
            ("veena", &VEENA, false),
            ("orpheus", &ORPHEUS, false),
            ("fish", &FISH, false),
            ("breeze", &BREEZE, true),
            ("maya2", &MAYA2, false),
            ("qwen-customvoice", &QWEN3_TTS, true),
        ] {
            let meta: PacketPipelines = serde_json::from_slice(
                &speech_pipeline_section(&mut model, spec, profile, 200000, &dir)
                    .unwrap()
                    .data,
            )
            .unwrap();
            check_tag_pipeline(name, &meta.pipelines[0], profile.languages, instructions);
        }
        assert_eq!(qwen_tags(false), SpeechTags::strip());
        assert_eq!(
            orpheus_tags(false).tags["giggle"].alias.as_deref(),
            Some("laugh")
        );
        assert_eq!(
            fish_tags(false).tags["laugh"].text.as_deref(),
            Some("[laughing]")
        );
        assert_eq!(
            breeze_tags(true).tags["sigh"].native_text(Some("zh")),
            Some("[叹气]")
        );
        assert_eq!(
            breeze_tags(true).tags["laugh"].native_text(Some("zh")),
            None
        );
        assert_eq!(
            qwen_tags(true).tags["rate"].instruction.as_deref(),
            Some("Speak at a {value} rate.")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn external_speaker_metadata_requires_a_new_driver_and_overlay() {
        let dir =
            std::env::temp_dir().join(format!("plow-speaker-contract-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let prompt = serde_json::json!({
            "text_template": "{input}", "instructions_template": null, "lead": 3, "drop": 5,
            "text_pad": 2, "text_bos": 3, "text_eos": 4, "none_code": 3072,
            "frame_codes": 16, "codec_language": [2154, 2156, 2157], "language_at": 2,
            "codec_auto": [2155, 2156, 2157], "codec_pad": 2148, "codec_bos": 2149,
            "languages": {"english": 2050}, "voices": {}, "dialect_languages": ["chinese", "auto"]
        });
        std::fs::write(dir.join(DUAL_STREAM), prompt.to_string()).unwrap();
        let config = serde_json::json!({"plow_frame_embed": {"conditioning": {
            "version": 1, "speaker_packet": "speaker_encoder.pkt", "reference_packet": "ref_encoder.pkt"
        }}});
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let mut tensors = Builder::new(2);
        let tokens = tensors.tensor("in.ids", 512 * 4);
        let positions = tensors.tensor("in.pos", 512 * 4);
        let kv_lengths = tensors.tensor("in.kvlen", 4);
        let overlay = tensors.tensor("in.encoder", 4 * 4);
        let overlay_index = tensors.tensor("in.encoder_index", 512 * 4);
        tensors.tensor("model.embed_tokens.weight", 4096 * 4 * 2);
        let mut model = Model {
            n_cu: 2,
            target: 0,
            tensors: tensors.tensors(),
            progs: (0..2).map(|_| Builder::new(2).finish()).collect(),
            kv_row_insts: Vec::new(),
            prog_t: vec![512, 1],
            gen: Vec::new(),
        };
        let spec = CausalPipelineSpec {
            name: "speech",
            max_context: 512,
            hidden: 4,
            decode_capacity: 1,
            overlay_rows: 1,
            ordered_dispatch: false,
            tensors: CausalPipelineTensors {
                tokens,
                positions,
                kv_lengths,
                overlay: Some(overlay),
                overlay_index: Some(overlay_index),
            },
        };
        let header = serde_json::json!({"model.embed_tokens.weight": {
            "dtype": "BF16", "shape": [4096, 4], "data_offsets": [0, 4096 * 4 * 2]
        }})
        .to_string()
        .into_bytes();
        let mut checkpoint = (header.len() as u64).to_le_bytes().to_vec();
        checkpoint.extend(header);
        let mut values = vec![0; 4096 * 4 * 2];
        values[16..24].copy_from_slice(&[0x80, 0x3f, 0, 0x40, 0x40, 0x40, 0x80, 0x40]);
        checkpoint.extend(values);
        std::fs::write(dir.join("model.safetensors"), checkpoint).unwrap();
        let mut emit = |spec| speech_pipeline_section(&mut model, spec, &QWEN3_TTS, 4096, &dir);
        let meta: PacketPipelines = serde_json::from_slice(&emit(spec).unwrap().data).unwrap();
        let pipe = &meta.pipelines[0];
        assert_eq!(pipe.driver, CONDITIONED_DRIVER);
        check_tag_pipeline("qwen-base", pipe, &[], false);
        assert_eq!(pipe.parameters["conditioning.text_pad"], 2);
        assert_eq!(
            pipe.tensors["conditioning.text_embeddings"].name,
            "model.embed_tokens.weight"
        );
        assert_eq!(
            pipe.tensors["conditioning.text_embeddings"].shape,
            [4096, 4]
        );
        assert_eq!(pipe.strings["prompt.voices"], "");
        assert_eq!(pipe.tensors["conditioning.text_pad_row"].shape, [4]);
        assert!(pipe.strings.contains_key(CONDITIONING_KEY));
        let mut no_overlay = spec;
        no_overlay.overlay_rows = 0;
        assert!(emit(no_overlay).is_err());
        let mut invalid = config.clone();
        invalid["plow_frame_embed"]["conditioning"]["version"] = 2.into();
        std::fs::write(dir.join("config.json"), invalid.to_string()).unwrap();
        assert!(emit(spec).is_err());
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        assert!(emit(spec).is_err());
        let mut preset = prompt;
        preset["voices"] = serde_json::json!({"ryan": {"code": 3061, "dialect": null}});
        std::fs::write(dir.join(DUAL_STREAM), preset.to_string()).unwrap();
        let legacy: PacketPipelines = serde_json::from_slice(&emit(spec).unwrap().data).unwrap();
        assert_eq!(legacy.pipelines[0].driver, DRIVER);
        assert!(!legacy.pipelines[0].strings.contains_key(CONDITIONING_KEY));
        assert!(!legacy.pipelines[0]
            .tensors
            .contains_key("conditioning.text_embeddings"));
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        assert!(emit(spec).is_err());
        assert!(model
            .tensors
            .iter()
            .find(|t| t.name == "model.embed_tokens.weight")
            .unwrap()
            .init
            .is_none());
        assert_eq!(
            model
                .tensors
                .iter()
                .find(|t| t.name == "in.prompt.text_pad")
                .unwrap()
                .init
                .as_deref(),
            Some([0x80, 0x3f, 0, 0x40, 0x40, 0x40, 0x80, 0x40].as_slice())
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
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
