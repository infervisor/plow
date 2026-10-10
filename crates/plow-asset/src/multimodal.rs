//! `plow.multimodal.v1`: how a text packet takes encoder rows (images, audio) in place of
//! placeholder tokens. Written by the emitter from the checkpoint and its processor config; read
//! by the runtime, which names no model.
//!
//! A prompt row whose token id has bit 31 set is a soft-token row: the LM's `Embed` gives it the
//! pad token's embedding (and per-layer input) and `MmRowsBf16` then replaces its hidden row with
//! the slab row the id names in `in.mm_table`. The runtime derives those ids from the media
//! content, so prefix-cache keys over token ids distinguish different images with the same text.
//!
//! A modality whose LM attends bidirectionally within each item (`attention =
//! "bidirectional_span"`) also sets [`SPAN_ID_BIT`] on its ids. Prefill derives each row's span
//! extent from the ids (`MmSpanExtent`), and the attention sites the emitter marked let a query
//! see the rest of its own run of span ids. A chunk end or sliding stage boundary must never fall
//! strictly inside such a run ([`span_safe_rows`]): the rows before it would miss the item's keys.
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const SECTION: &str = "plow.multimodal.v1";
pub const VERSION: u32 = 1;
/// Soft-token ids carry this bit; the low 31 bits are a content hash.
pub const ROW_ID_BIT: u32 = 0x8000_0000;
/// Soft-token ids of a [`MediaAttention::BidirectionalSpan`] modality also carry this bit (no
/// other modality's do); the low 30 bits are then the content hash.
pub const SPAN_ID_BIT: u32 = 0x4000_0000;
/// The LM tensors the soft-token rows travel through.
pub const SLAB_TENSOR: &str = "in.mm_slab";
pub const TABLE_TENSOR: &str = "in.mm_table";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MmContract {
    pub version: u32,
    /// LM hidden width of one encoder row (bf16 in the slab).
    pub hidden: u32,
    /// Token id soft-token rows embed as before their row is replaced.
    pub pad_token: u32,
    /// Rows of `in.mm_slab` (encoder rows of in-flight prompts).
    pub slab_rows: u32,
    /// Entries of `in.mm_table` (`(id, slab row)` pairs, a power of two).
    pub table_capacity: u32,
    pub modalities: Vec<MmModality>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MmModality {
    /// `image` or `audio`: the request content parts this modality serves.
    pub kind: String,
    /// Encoder sidecar beside the text packet (`forward.v1` pipeline `mm.encode`).
    pub packet: String,
    /// The token the chat template renders per item; replaced by `begin, rows.., end`.
    pub placeholder: u32,
    pub begin: Option<u32>,
    pub end: Option<u32>,
    /// Host preprocessing algorithm, parameterized by `parameters` (`*_f32` keys are f32 bits).
    pub processor: String,
    pub parameters: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "MediaAttention::is_causal")]
    pub attention: MediaAttention,
}

/// How the LM's prompt attention treats one item's soft-token rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaAttention {
    #[default]
    Causal,
    /// Rows of one item also attend to the item's later rows, on the attention sites the emitter
    /// marked. Items are wrapped by `begin`/`end`, so a run of consecutive span ids is one item.
    BidirectionalSpan,
}

impl MediaAttention {
    pub fn is_causal(&self) -> bool {
        *self == MediaAttention::Causal
    }
}

/// `true` for a soft-token id of a bidirectional-span modality.
pub fn is_span_id(id: u32) -> bool {
    id & (ROW_ID_BIT | SPAN_ID_BIT) == ROW_ID_BIT | SPAN_ID_BIT
}

/// The rows of `prompt[start..start + rows]` one prefill may take so that neither its end nor an
/// internal sliding stage boundary (every `stage` rows from `start`; 0 = unstaged) falls strictly
/// inside a run of span ids. Cuts back to the start of the first run that would be split: 0 when
/// that run begins at `start` (the request then needs a share of at least the run).
pub fn span_safe_rows(prompt: &[u32], start: usize, rows: usize, stage: usize) -> usize {
    let end = (start + rows).min(prompt.len());
    let mut i = start;
    while i < end {
        if !is_span_id(prompt[i]) {
            i += 1;
            continue;
        }
        let s = i;
        while i < prompt.len() && is_span_id(prompt[i]) {
            i += 1;
        }
        let split_end = end < i;
        let split_stage = stage > 0 && start + ((s - start) / stage + 1) * stage < i.min(end);
        if split_end || split_stage {
            return s - start;
        }
    }
    end - start
}

/// As [`span_safe_rows`], but a run that begins at `start` and does not fit is taken whole when
/// it fits `limit` rows (a request must make progress past its own span).
pub fn span_chunk_rows(prompt: &[u32], start: usize, rows: usize, limit: usize, stage: usize) -> usize {
    match span_safe_rows(prompt, start, rows, stage) {
        0 => {
            let run = prompt[start..].iter().take_while(|&&id| is_span_id(id)).count();
            if run <= limit { run } else { 0 }
        }
        safe => safe,
    }
}

/// A host preprocessing algorithm the runtime implements ([`MmModality::processor`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Processor {
    /// Images: aspect-preserving bicubic resize, then `patch_size`² patches pooled `pool`².
    AspectPatches,
    /// Audio: semicausal log-mel spectrogram for a conformer tower.
    SemicausalLogMel,
    /// Audio: raw `frame_samples`-sample frames for an encoder-free embedder.
    WaveformFrames,
}

/// (processor, name, modality kind, required parameters, optional parameters).
type ProcessorSpec = (Processor, &'static str, &'static str, &'static [&'static str], &'static [&'static str]);

impl Processor {
    /// Every implemented processor. A contract naming anything else, or a parameter outside these
    /// lists, is refused: ignoring it would preprocess differently from the checkpoint's processor.
    pub const ALL: [ProcessorSpec; 3] = [
        (
            Processor::AspectPatches,
            "aspect_patches",
            "image",
            &["patch_size", "pool", "max_soft_tokens"],
            &["rescale_f32", "resample", "normalize", "mean0_f32", "mean1_f32", "mean2_f32", "std0_f32", "std1_f32", "std2_f32"],
        ),
        (
            Processor::SemicausalLogMel,
            "semicausal_log_mel",
            "audio",
            &[
                "sample_rate",
                "frame_length",
                "hop_length",
                "fft_length",
                "mel_bins",
                "min_frequency_f32",
                "max_frequency_f32",
                "mel_floor_f32",
            ],
            &["pad_multiple", "max_samples", "subsample", "max_soft_tokens"],
        ),
        (Processor::WaveformFrames, "waveform_frames", "audio", &["sample_rate", "frame_samples", "max_soft_tokens"], &[]),
    ];

    pub fn name(self) -> &'static str {
        Self::ALL.iter().find(|p| p.0 == self).map_or("", |p| p.1)
    }
}

/// PIL's `Image.BICUBIC`, the only resample filter `aspect_patches` implements.
pub const RESAMPLE_BICUBIC: u64 = 3;

impl MmModality {
    pub fn param(&self, key: &str) -> Option<u64> {
        self.parameters.get(key).copied()
    }

    pub fn param_f32(&self, key: &str) -> Option<f32> {
        self.param(key).and_then(|v| u32::try_from(v).ok()).map(f32::from_bits)
    }

    /// The processor this modality names, with its parameters checked. `Err` only on a contract
    /// [`MmContract::validate`] refuses.
    pub fn processor(&self) -> Result<Processor, String> {
        let known = |kind: &str| Processor::ALL.iter().filter(|p| p.2 == kind).map(|p| p.1).collect::<Vec<_>>().join(", ");
        let Some(&(p, _, kind, required, optional)) = Processor::ALL.iter().find(|p| p.1 == self.processor) else {
            return Err(format!(
                "{} processor {:?} is not implemented by this runtime (it knows: {}); upgrade plowrt, or rebuild the \
                 packet with a plowc whose emitter names a known processor",
                self.kind,
                self.processor,
                known(&self.kind)
            ));
        };
        if kind != self.kind {
            return Err(format!(
                "processor {:?} serves {kind} parts, not {} (known {} processors: {}); rebuild the packet",
                self.processor,
                self.kind,
                self.kind,
                known(&self.kind)
            ));
        }
        if let Some(key) = required.iter().find(|k| !self.parameters.contains_key(**k)) {
            return Err(format!(
                "{} processor {:?} lacks required parameter {key:?}; rebuild the packet with a plowc that writes it",
                self.kind, self.processor
            ));
        }
        if let Some(key) = self.parameters.keys().find(|k| !required.contains(&k.as_str()) && !optional.contains(&k.as_str())) {
            return Err(format!(
                "{} processor {:?} has parameter {key:?}, which this runtime does not implement; upgrade plowrt",
                self.kind, self.processor
            ));
        }
        if let Some(r) = self.param("resample").filter(|&r| p == Processor::AspectPatches && r != RESAMPLE_BICUBIC) {
            return Err(format!(
                "image resample filter {r} is not implemented (only PIL bicubic, {RESAMPLE_BICUBIC}); upgrade plowrt"
            ));
        }
        let positive: &[&str] = match p {
            Processor::AspectPatches => &["patch_size", "pool", "max_soft_tokens"],
            Processor::SemicausalLogMel => &["sample_rate", "frame_length", "hop_length", "fft_length", "mel_bins"],
            Processor::WaveformFrames => &["sample_rate", "frame_samples", "max_soft_tokens"],
        };
        if let Some(key) = positive.iter().find(|k| self.param(k) == Some(0)) {
            return Err(format!("{} processor {:?} has {key} = 0; rebuild the packet", self.kind, self.processor));
        }
        Ok(p)
    }
}

impl MmContract {
    /// Parse and validate a `plow.multimodal.v1` section. The version is read before the body, so
    /// a newer contract fails on its version, not on a field this runtime does not know.
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        #[derive(Deserialize)]
        struct Version {
            version: Option<u32>,
        }
        let v: Version = serde_json::from_slice(bytes).map_err(|e| format!("multimodal contract: {e}"))?;
        match v.version {
            Some(VERSION) => {}
            Some(other) => return Err(unsupported_version(other)),
            None => return Err("multimodal contract has no version; rebuild the packet".into()),
        }
        let c: Self = serde_json::from_slice(bytes).map_err(|e| format!("multimodal contract: {e}"))?;
        c.validate()?;
        Ok(c)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version != VERSION {
            return Err(unsupported_version(self.version));
        }
        let bad = |what: String| Err(format!("invalid multimodal contract: {what}; rebuild the packet"));
        if self.hidden == 0 || self.slab_rows == 0 {
            return bad(format!("hidden {} and slab_rows {} must be positive", self.hidden, self.slab_rows));
        }
        if !self.table_capacity.is_power_of_two() || self.table_capacity < self.slab_rows {
            return bad(format!("table_capacity {} must be a power of two >= slab_rows {}", self.table_capacity, self.slab_rows));
        }
        if self.modalities.is_empty() {
            return bad("no modalities".into());
        }
        for (i, m) in self.modalities.iter().enumerate() {
            if !matches!(m.kind.as_str(), "image" | "audio") {
                return bad(format!("modality kind {:?} (known: image, audio)", m.kind));
            }
            if !m.attention.is_causal() && (m.begin.is_none() || m.end.is_none()) {
                return bad(format!("the {} modality attends within spans but has no begin/end tokens", m.kind));
            }
            if self.modalities[..i].iter().any(|o| o.kind == m.kind) {
                return bad(format!("two {} modalities", m.kind));
            }
            if self.modalities[..i].iter().any(|o| o.placeholder == m.placeholder) {
                return bad(format!("placeholder {} serves two modalities", m.placeholder));
            }
            if m.packet.is_empty() {
                return bad(format!("the {} modality names no encoder packet", m.kind));
            }
            if [Some(m.placeholder), m.begin, m.end].into_iter().flatten().any(|t| t & ROW_ID_BIT != 0) {
                return bad(format!("{} placeholder/begin/end ids may not set bit 31", m.kind));
            }
            m.processor()?;
        }
        Ok(())
    }

    pub fn modality(&self, kind: &str) -> Option<&MmModality> {
        self.modalities.iter().find(|m| m.kind == kind)
    }

    pub fn has_spans(&self) -> bool {
        self.modalities.iter().any(|m| !m.attention.is_causal())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: u32 = ROW_ID_BIT | SPAN_ID_BIT | 7;
    const A: u32 = ROW_ID_BIT | 7;

    #[test]
    fn span_ids_are_tagged_runs() {
        assert!(is_span_id(S));
        assert!(!is_span_id(A));
        assert!(!is_span_id(SPAN_ID_BIT));
    }

    #[test]
    fn chunks_never_end_inside_a_span() {
        // text 0..4, span 4..8, text 8..10
        let p = [1, 2, 3, 4, S, S, S, S, 5, 6];
        assert_eq!(span_safe_rows(&p, 0, 10, 0), 10);
        assert_eq!(span_safe_rows(&p, 0, 6, 0), 4, "cut back to the span start");
        assert_eq!(span_safe_rows(&p, 0, 8, 0), 8, "ending at the span end is safe");
        assert_eq!(span_safe_rows(&p, 0, 4, 0), 4);
        assert_eq!(span_safe_rows(&p, 4, 2, 0), 0, "a span at the start needs the whole run");
        assert_eq!(span_safe_rows(&p, 5, 5, 0), 5, "a prefix-cache hit inside the span continues it");
        // Causal media ids never constrain.
        let q = [1, A, A, A, 2];
        assert_eq!(span_safe_rows(&q, 0, 2, 0), 2);
        // A span at the start is taken whole when the request may hold it.
        assert_eq!(span_chunk_rows(&p, 4, 2, 4096, 0), 4);
        assert_eq!(span_chunk_rows(&p, 4, 2, 3, 0), 0);
        assert_eq!(span_chunk_rows(&p, 0, 6, 4096, 0), 4);
    }

    #[test]
    fn stage_boundaries_never_split_a_span() {
        // stage 4: boundaries at 4 and 8; span 3..6 crosses 4.
        let p = [1, 2, 3, S, S, S, 4, 5, 6, 7, 8, 9];
        assert_eq!(span_safe_rows(&p, 0, 12, 4), 3);
        // A span starting on the boundary is safe.
        let q = [1, 2, 3, 4, S, S, 5, 6, 7, 8];
        assert_eq!(span_safe_rows(&q, 0, 10, 4), 10);
        // Stage boundaries count from the chunk start, not position 0.
        assert_eq!(span_safe_rows(&p, 3, 9, 4), 9);
        assert_eq!(span_safe_rows(&p, 1, 11, 4), 2, "boundary at 5 splits 3..6");
        // A span past the chunk end does not matter.
        let r = [1, 2, 3, 4, 5, 6, S, S];
        assert_eq!(span_safe_rows(&r, 0, 6, 4), 6);
    }

    #[test]
    fn causal_contracts_serialize_without_the_attention_field() {
        let m = MmModality {
            kind: "audio".into(),
            packet: "a.pkt".into(),
            placeholder: 3,
            begin: None,
            end: None,
            processor: "p".into(),
            parameters: BTreeMap::new(),
            attention: MediaAttention::Causal,
        };
        assert!(!serde_json::to_string(&m).unwrap().contains("attention"));
        let span = MmModality { attention: MediaAttention::BidirectionalSpan, ..m };
        assert!(serde_json::to_string(&span).unwrap().contains("\"attention\":\"bidirectional_span\""));
    }
}

fn unsupported_version(v: u32) -> String {
    format!(
        "multimodal contract version {v} is not supported (this runtime reads version {VERSION}); serve the packet \
         with a plowrt that reads it, or rebuild it with a matching plowc"
    )
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    fn params(kv: &[(&str, u64)]) -> BTreeMap<String, u64> {
        kv.iter().map(|&(k, v)| (k.to_string(), v)).collect()
    }

    fn image() -> MmModality {
        MmModality {
            kind: "image".into(),
            packet: "mm_vision.pkt".into(),
            placeholder: 7,
            begin: Some(8),
            end: Some(9),
            processor: "aspect_patches".into(),
            parameters: params(&[("patch_size", 16), ("pool", 3), ("max_soft_tokens", 280), ("resample", 3)]),
            attention: MediaAttention::Causal,
        }
    }

    fn frames() -> MmModality {
        MmModality {
            kind: "audio".into(),
            packet: "mm_audio.pkt".into(),
            placeholder: 10,
            begin: None,
            end: None,
            processor: "waveform_frames".into(),
            parameters: params(&[("sample_rate", 16_000), ("frame_samples", 640), ("max_soft_tokens", 750)]),
            attention: MediaAttention::Causal,
        }
    }

    fn contract(modalities: Vec<MmModality>) -> MmContract {
        MmContract { version: VERSION, hidden: 4, pad_token: 0, slab_rows: 8, table_capacity: 16, modalities }
    }

    fn refused(c: &MmContract, needle: &str) {
        let err = c.validate().unwrap_err();
        assert!(err.contains(needle), "{err:?} lacks {needle:?}");
    }

    #[test]
    fn known_processors_validate_and_dispatch() {
        let c = contract(vec![image(), frames()]);
        c.validate().unwrap();
        assert_eq!(c.modality("image").unwrap().processor(), Ok(Processor::AspectPatches));
        assert_eq!(c.modality("audio").unwrap().processor(), Ok(Processor::WaveformFrames));
        assert_eq!(Processor::SemicausalLogMel.name(), "semicausal_log_mel");
    }

    #[test]
    fn unknown_or_mislabelled_processors_are_refused() {
        let mut m = frames();
        m.processor = "log_mel_v2".into();
        refused(&contract(vec![m]), "not implemented by this runtime (it knows: semicausal_log_mel, waveform_frames)");
        let mut m = image();
        m.processor = "waveform_frames".into();
        refused(&contract(vec![m]), "serves audio parts, not image");
        let mut m = image();
        m.processor = "squash_patches".into();
        refused(&contract(vec![m]), "it knows: aspect_patches");
    }

    #[test]
    fn missing_unknown_and_unsupported_parameters_are_refused() {
        let mut m = frames();
        m.parameters.remove("frame_samples");
        refused(&contract(vec![m]), "lacks required parameter \"frame_samples\"");
        let mut m = frames();
        m.parameters.insert("preemphasis_f32".into(), 1);
        refused(&contract(vec![m]), "parameter \"preemphasis_f32\", which this runtime does not implement");
        let mut m = image();
        m.parameters.insert("resample".into(), 2);
        refused(&contract(vec![m]), "only PIL bicubic");
        let mut m = image();
        m.parameters.insert("pool".into(), 0);
        refused(&contract(vec![m]), "pool = 0");
    }

    #[test]
    fn structure_is_checked() {
        refused(&contract(vec![image(), image()]), "two image modalities");
        let mut a = frames();
        a.placeholder = 7;
        refused(&contract(vec![image(), a]), "placeholder 7 serves two modalities");
        let mut a = frames();
        a.end = Some(ROW_ID_BIT | 3);
        refused(&contract(vec![a]), "may not set bit 31");
        let mut c = contract(vec![image()]);
        c.table_capacity = 12;
        refused(&c, "power of two");
        refused(&contract(vec![]), "no modalities");
    }

    #[test]
    fn versions_are_read_before_the_body() {
        let c = contract(vec![image(), frames()]);
        assert_eq!(MmContract::from_json(&serde_json::to_vec(&c).unwrap()).unwrap(), c);
        let mut v2 = serde_json::to_value(&c).unwrap();
        v2["version"] = 2.into();
        v2["new_field"] = true.into();
        let err = MmContract::from_json(&serde_json::to_vec(&v2).unwrap()).unwrap_err();
        assert!(err.contains("version 2 is not supported (this runtime reads version 1)"), "{err}");
        let mut c0 = c.clone();
        c0.version = 0;
        assert!(c0.validate().unwrap_err().contains("version 0"));
        assert!(MmContract::from_json(b"{}").unwrap_err().contains("no version"));
    }
}
