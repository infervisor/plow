//! The `media_geometry.v1` obligation (lean-plow `Plow/MediaGeometry.lean`), derived from a
//! bundle's packet pipeline sections, tensor shapes and contracts. A missing parameter, a
//! referenced sidecar that is absent, or a shape product that overflows rejects the bundle.

use serde_json::{json, Value};

use crate::multimodal::MmContract;
use crate::packet_pipeline::PacketPipeline;

pub const ENDPOINT: &str = "media_geometry.v1";

/// One packet of a bundle, as qualification sees it.
pub struct MediaPacket<'a> {
    pub file: &'a str,
    pub pipelines: &'a [PacketPipeline],
    /// Pieces in `asr_vocabulary.json`, when the packet has one.
    pub vocabulary: Option<usize>,
    pub multimodal: Option<&'a MmContract>,
    /// Bytes of `in.mm_slab` and `in.mm_table`, when the packet has them.
    pub mm_tensors: Option<(u64, u64)>,
}

fn param(p: &PacketPipeline, name: &str) -> Result<u64, String> {
    p.parameters.get(name).copied().ok_or_else(|| format!("{}: parameter {name:?} is missing", p.name))
}

fn elements(p: &PacketPipeline, role: &str) -> Result<u64, String> {
    let t = p.tensors.get(role).ok_or_else(|| format!("{}: tensor {role:?} is missing", p.name))?;
    t.shape.iter().try_fold(1u64, |n, &d| n.checked_mul(d)).ok_or_else(|| format!("{}: {role:?} overflows", p.name))
}

fn listed(p: &PacketPipeline, prefix: &str) -> Result<Vec<u64>, String> {
    (0..param(p, &format!("{prefix}.count"))?).map(|i| param(p, &format!("{prefix}.{i}"))).collect()
}

fn sidecar<'a>(
    main: &PacketPipeline,
    key: &str,
    sidecars: &'a [MediaPacket<'a>],
    pick: impl Fn(&PacketPipeline) -> bool,
) -> Result<&'a PacketPipeline, String> {
    let file = main.strings.get(key).ok_or_else(|| format!("{}: string {key:?} is missing", main.name))?;
    sidecars
        .iter()
        .find(|s| s.file == file)
        .ok_or_else(|| format!("{}: sidecar {file} is absent", main.name))?
        .pipelines
        .iter()
        .find(|p| pick(p))
        .ok_or_else(|| format!("{file}: no matching pipeline for {}", main.name))
}

fn audio_lm(p: &PacketPipeline, sidecars: &[MediaPacket<'_>]) -> Result<Value, String> {
    let e = sidecar(p, "encoder.packet", sidecars, |e| e.name == "audio.encode")?;
    Ok(json!({"kind": "audio_lm",
        "sample_rate": param(p, "audio.sample_rate")?, "max_seconds": param(p, "audio.max_seconds")?,
        "encoder_sample_rate": param(e, "audio.sample_rate")?, "encoder_max_samples": param(e, "audio.max_samples")?,
        "hop": param(e, "audio.frontend.hop")?, "chunk_frames": param(e, "input.chunk_frames")?,
        "frame_stride": param(e, "encoder.frame_stride")?, "overlay_rows": param(p, "overlay_rows")?,
        "encoder_output_rows": param(e, "output_rows")?, "hidden": param(p, "hidden")?,
        "encoder_width": param(e, "output_width")?, "max_context": param(p, "max_context")?,
        "max_tokens": param(p, "output.max_tokens")?, "reserve_per_row": param(p, "output.reserve_per_row")?,
        "reserve_extra": param(p, "output.reserve_extra")?, "audio_token": param(p, "audio.token_id")?,
        "stops": listed(p, "stop")?}))
}

fn codec_lm(p: &PacketPipeline, sidecars: &[MediaPacket<'_>]) -> Result<Value, String> {
    let c = sidecar(p, "codec.packet", sidecars, |c| c.driver == "codec.v1")?;
    let prompt = param(p, "prompt.prefix.count")?
        .checked_add(param(p, "prompt.suffix.count")?)
        .and_then(|n| n.checked_add(u64::from(p.strings.contains_key("prompt.voice_token"))))
        .ok_or("codec LM: prompt length overflows")?;
    Ok(json!({"kind": "codec_lm",
        "lm_sample_rate": param(p, "audio.sample_rate")?, "lm_codebook": param(p, "codec.codebook")?,
        "lm_frame_codes": param(p, "codec.frame_codes")?, "lm_frame_samples": param(p, "codec.frame_samples")?,
        "token_base": param(p, "audio.token_base")?, "max_new_tokens": param(p, "tokens.max_new_cap")?,
        "prompt_tokens": prompt, "max_context": param(p, "max_context")?, "stops": listed(p, "stop")?,
        "codec_sample_rate": param(c, "audio.sample_rate")?, "codebook": param(c, "codec.codebook")?,
        "frame_codes": param(c, "codec.frame_codes")?, "frame_samples": param(c, "codec.frame_samples")?,
        "codes_capacity": elements(c, "codes")?, "pcm_capacity": elements(c, "pcm")?,
        "window_frames": param(c, "stream.window_frames")?,
        "lookahead_frames": param(c, "stream.lookahead_frames")?}))
}

fn guided_lm(p: &PacketPipeline) -> Result<Value, String> {
    Ok(json!({"kind": "guided_lm",
        "speech_vocab": param(p, "lm.speech_vocab")?, "start_speech": param(p, "lm.start_speech")?,
        "stop_speech": param(p, "lm.stop_speech")?, "valid_below": param(p, "lm.valid_below")?,
        "text_vocab": param(p, "lm.text_vocab")?, "start_text": param(p, "lm.start_text")?,
        "stop_text": param(p, "lm.stop_text")?, "max_speech_tokens": param(p, "lm.max_speech_tokens")?,
        "speech_positions": param(p, "lm.speech_pos_rows")?, "overlay_rows": param(p, "overlay_rows")?,
        "max_context": param(p, "max_context")?}))
}

fn rnnt(p: &PacketPipeline, vocabulary: Option<usize>) -> Result<Value, String> {
    let transforms = (0..param(p, "frame_transform.count")?)
        .map(|i| -> Result<Value, String> {
            let f = |k: &str| param(p, &format!("frame_transform.{i}.{k}"));
            Ok(json!({"kernel": f("kernel")?, "stride": f("stride")?,
                "pad_before": f("pad_before")?, "pad_after": f("pad_after")?}))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let joint = p.tensors.get("encoder.joint").ok_or("rnnt: tensor \"encoder.joint\" is missing")?;
    Ok(json!({"kind": "rnnt",
        "hop": param(p, "audio.frontend.hop")?, "max_samples": param(p, "audio.max_samples")?,
        "input_frames": param(p, "input_frames")?, "transforms": transforms, "frames": param(p, "frames")?,
        "joint_rows": joint.shape.first().copied().ok_or("rnnt: empty joint shape")?,
        "blank": param(p, "blank_id")?, "vocab": vocabulary.ok_or("rnnt: no asr_vocabulary.json")?,
        "max_symbols_per_frame": param(p, "max_symbols_per_frame")?}))
}

fn vad(p: &PacketPipeline) -> Result<Value, String> {
    let banks = param(p, "state_banks")?;
    let states = (0..banks.checked_mul(2).ok_or("vad: state banks overflow")?)
        .map(|i| elements(p, &format!("state.{i}")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({"kind": "vad",
        "frame_samples": param(p, "frame_samples")?, "context_samples": param(p, "context_samples")?,
        "input_elements": elements(p, "input")?, "state_banks": banks, "state_sizes": states,
        "min_speech_ms": param(p, "policy.min_speech_ms")?, "max_duration_ms": param(p, "bounds.max_duration_ms")?}))
}

fn multimodal(mm: &MmContract, tensors: Option<(u64, u64)>, sidecars: &[MediaPacket<'_>]) -> Result<Value, String> {
    let (slab_bytes, table_bytes) = tensors.ok_or("multimodal: in.mm_slab/in.mm_table are missing")?;
    let mut tokens = Vec::new();
    let mut widths = Vec::new();
    for m in &mm.modalities {
        tokens.push(u64::from(m.placeholder));
        tokens.extend(m.begin.map(u64::from));
        tokens.extend(m.end.map(u64::from));
        let encoder = sidecars
            .iter()
            .find(|s| s.file == m.packet)
            .ok_or_else(|| format!("multimodal: sidecar {} is absent", m.packet))?;
        let output = encoder
            .pipelines
            .iter()
            .find_map(|p| p.tensors.get("output"))
            .ok_or_else(|| format!("{}: no output tensor", m.packet))?;
        widths.push(*output.shape.last().ok_or("multimodal: empty output shape")?);
    }
    let width = widths[0];
    if widths.iter().any(|&w| w != width) {
        return Err("multimodal: encoder output widths differ".into());
    }
    Ok(json!({"kind": "multimodal", "hidden": mm.hidden, "lm_hidden": width, "pad_token": mm.pad_token,
        "tokens": tokens, "slab_rows": mm.slab_rows, "table_capacity": mm.table_capacity,
        "slab_bytes": slab_bytes, "table_bytes": table_bytes}))
}

/// The bundle's media obligation, or `None` when `main` has no speech or multimodal pipeline.
pub fn request(main: &MediaPacket<'_>, sidecars: &[MediaPacket<'_>]) -> Result<Option<Value>, String> {
    let mut families = Vec::new();
    for p in main.pipelines {
        let family = match p.driver.as_str() {
            "causal.v1" if p.parameters.get("overlay_rows").is_some_and(|&r| r > 0) => audio_lm(p, sidecars)?,
            "tts.codec_lm.v1" => codec_lm(p, sidecars)?,
            "tts.guided_lm.v1" => guided_lm(p)?,
            "rnnt.greedy.v1" => rnnt(p, main.vocabulary)?,
            "vad.frame.v1" => vad(p)?,
            _ => continue,
        };
        families.push(family);
    }
    if let Some(mm) = main.multimodal {
        families.push(multimodal(mm, main.mm_tensors, sidecars)?);
    }
    Ok((!families.is_empty()).then(|| json!({"schema": 1, "families": families})))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet_pipeline::{PipelineDType, PipelineTensor};
    use std::collections::BTreeMap;

    fn pipeline(name: &str, driver: &str, params: &[(&str, u64)], strings: &[(&str, &str)],
        tensors: &[(&str, Vec<u64>)]) -> PacketPipeline {
        PacketPipeline {
            name: name.into(),
            driver: driver.into(),
            programs: BTreeMap::from([("forward".into(), 0)]),
            tensors: tensors.iter().map(|(role, shape)| (role.to_string(),
                PipelineTensor { name: format!("t.{role}"), dtype: PipelineDType::U32, shape: shape.clone() })).collect(),
            parameters: params.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            strings: strings.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        }
    }

    fn codec_lm() -> (Vec<PacketPipeline>, Vec<PacketPipeline>) {
        let lm = pipeline("speech", "tts.codec_lm.v1", &[("audio.sample_rate", 24000), ("codec.codebook", 4096),
            ("codec.frame_codes", 7), ("codec.frame_samples", 2048), ("audio.token_base", 128266),
            ("tokens.max_new_cap", 1400), ("prompt.prefix.count", 1), ("prompt.suffix.count", 3),
            ("max_context", 2048), ("stop.count", 1), ("stop.0", 128258)],
            &[("codec.packet", "codec.pkt"), ("prompt.voice_token", "<v>")], &[]);
        let codec = pipeline("codec.decode", "codec.v1", &[("audio.sample_rate", 24000), ("codec.codebook", 4096),
            ("codec.frame_codes", 7), ("codec.frame_samples", 2048), ("stream.window_frames", 6),
            ("stream.lookahead_frames", 2)], &[], &[("codes", vec![3584]), ("pcm", vec![1024, 1024])]);
        (vec![lm], vec![codec])
    }

    fn packet<'a>(file: &'a str, pipelines: &'a [PacketPipeline]) -> MediaPacket<'a> {
        MediaPacket { file, pipelines, vocabulary: None, multimodal: None, mm_tensors: None }
    }

    #[test]
    fn codec_lm_request_reads_both_packets() {
        let (lm, codec) = codec_lm();
        let sidecars = [packet("model.pkt", &lm), packet("codec.pkt", &codec)];
        let request = request(&packet("model.pkt", &lm), &sidecars).unwrap().unwrap();
        let family = &request["families"][0];
        assert_eq!(family["kind"], "codec_lm");
        assert_eq!(family["prompt_tokens"], 5);
        assert_eq!(family["codes_capacity"], 3584);
        assert_eq!(family["pcm_capacity"], 1048576);
        assert_eq!(family["stops"], serde_json::json!([128258]));
    }

    #[test]
    fn missing_parameters_sidecars_and_overflow_reject() {
        let (mut lm, codec) = codec_lm();
        let sidecars = [packet("codec.pkt", &codec)];
        assert!(request(&packet("model.pkt", &lm), &[]).is_err(), "absent sidecar");
        lm[0].parameters.remove("tokens.max_new_cap");
        assert!(request(&packet("model.pkt", &lm), &sidecars).is_err(), "missing parameter");
        let text = vec![pipeline("decode", "causal.v1", &[("overlay_rows", 0)], &[], &[])];
        assert!(request(&packet("model.pkt", &text), &[]).unwrap().is_none(), "plain LM owes nothing");
        let mut wide = codec.clone();
        wide[0].tensors.get_mut("pcm").unwrap().shape = vec![u64::MAX, 2];
        let (lm, _) = codec_lm();
        assert!(request(&packet("model.pkt", &lm), &[packet("codec.pkt", &wide)]).is_err(), "overflow");
    }
}
