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
    p.parameters
        .get(name)
        .copied()
        .ok_or_else(|| format!("{}: parameter {name:?} is missing", p.name))
}

fn elements(p: &PacketPipeline, role: &str) -> Result<u64, String> {
    let t = p
        .tensors
        .get(role)
        .ok_or_else(|| format!("{}: tensor {role:?} is missing", p.name))?;
    t.shape
        .iter()
        .try_fold(1u64, |n, &d| n.checked_mul(d))
        .ok_or_else(|| format!("{}: {role:?} overflows", p.name))
}

fn listed(p: &PacketPipeline, prefix: &str) -> Result<Vec<u64>, String> {
    (0..param(p, &format!("{prefix}.count"))?)
        .map(|i| param(p, &format!("{prefix}.{i}")))
        .collect()
}

fn sidecar<'a>(
    main: &PacketPipeline,
    key: &str,
    sidecars: &'a [MediaPacket<'a>],
    pick: impl Fn(&PacketPipeline) -> bool,
) -> Result<&'a PacketPipeline, String> {
    let file = main
        .strings
        .get(key)
        .ok_or_else(|| format!("{}: string {key:?} is missing", main.name))?;
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
    // A depth-stage LM emits one token per frame; a codec-token LM one per code.
    let codebook = param(p, "codec.codebook")?;
    let audio_tokens = if p.parameters.get("frames.depth").copied().unwrap_or(0) != 0 {
        codebook
    } else {
        codebook
            .checked_mul(param(p, "codec.frame_codes")?)
            .ok_or("codec LM: audio token range overflows")?
    };
    let mut family = json!({"kind": "codec_lm",
        "lm_sample_rate": param(p, "audio.sample_rate")?, "lm_codebook": param(p, "codec.codebook")?,
        "lm_frame_codes": param(p, "codec.frame_codes")?, "lm_frame_samples": param(p, "codec.frame_samples")?,
        "token_base": param(p, "audio.token_base")?, "audio_tokens": audio_tokens,
        "max_new_tokens": param(p, "tokens.max_new_cap")?,
        "prompt_tokens": prompt, "max_context": param(p, "max_context")?, "stops": listed(p, "stop")?,
        "codec_sample_rate": param(c, "audio.sample_rate")?, "codebook": param(c, "codec.codebook")?,
        "frame_codes": param(c, "codec.frame_codes")?, "frame_samples": param(c, "codec.frame_samples")?,
        "codes_capacity": elements(c, "codes")?, "pcm_capacity": elements(c, "pcm")?,
        "window_frames": param(c, "stream.window_frames")?,
        "lookahead_frames": param(c, "stream.lookahead_frames")?,
        "stream_slots": 0, "step_frames": 0, "steps": [], "slot_capacity": 0, "pos_capacity": 0,
        "rope_capacity": 0, "new_rows_capacities": [], "clone_frames": [], "clone_pcm_capacity": 0,
        "clone_codes_capacity": 0, "clone_sample_rate": 0, "clone_frame_samples": 0,
        "clone_frame_codes": 0, "clone_codebook": 0, "clone_eos_code": 0});
    if c.parameters.get("stream.stateful").copied().unwrap_or(0) != 0 {
        family.as_object_mut().unwrap().extend(stateful_codec(c)?);
    }
    if p.strings.contains_key("clone.encoder") {
        let e = sidecar(p, "clone.encoder", sidecars, |e| {
            e.driver == "codec.encode.v1"
        })?;
        let frames: Vec<u64> = e
            .programs
            .keys()
            .filter_map(|role| role.strip_prefix("encode.f")?.parse().ok())
            .collect();
        if frames.is_empty() {
            return Err("clone.encoder has no encode.f{F} program".into());
        }
        let Value::Object(m) = json!({"clone_frames": frames,
            "clone_pcm_capacity": elements(e, "pcm")?, "clone_codes_capacity": elements(e, "codes")?,
            "clone_sample_rate": param(e, "audio.sample_rate")?,
            "clone_frame_samples": param(e, "codec.frame_samples")?,
            "clone_frame_codes": param(e, "codec.frame_codes")?,
            "clone_codebook": param(e, "codec.codebook")?, "clone_eos_code": param(p, "clone.eos_code")?})
        else {
            unreachable!()
        };
        family.as_object_mut().unwrap().extend(m);
    }
    Ok(family)
}

/// The stateful stream steps `step.b{B}.f{N}` and the inputs they index.
fn stateful_codec(c: &PacketPipeline) -> Result<serde_json::Map<String, Value>, String> {
    let steps: Vec<[u64; 2]> = c
        .programs
        .keys()
        .filter_map(|role| {
            let (b, n) = role.strip_prefix("step.b")?.split_once(".f")?;
            Some([b.parse().ok()?, n.parse().ok()?])
        })
        .collect();
    let new_rows = (0..param(c, "lengths.count")?)
        .map(|k| elements(c, &format!("stream.n.{k}")))
        .collect::<Result<Vec<_>, _>>()?;
    let Value::Object(m) = json!({"stream_slots": param(c, "stream.slots")?,
        "step_frames": param(c, "stream.step_frames")?, "steps": steps,
        "slot_capacity": elements(c, "stream.slot")?, "pos_capacity": elements(c, "stream.pos")?,
        "rope_capacity": elements(c, "stream.rope_pos")?, "new_rows_capacities": new_rows})
    else {
        unreachable!()
    };
    Ok(m)
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
    let joint = p
        .tensors
        .get("encoder.joint")
        .ok_or("rnnt: tensor \"encoder.joint\" is missing")?;
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

fn multimodal(
    mm: &MmContract,
    tensors: Option<(u64, u64)>,
    sidecars: &[MediaPacket<'_>],
) -> Result<Value, String> {
    let (slab_bytes, table_bytes) =
        tensors.ok_or("multimodal: in.mm_slab/in.mm_table are missing")?;
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
        widths.push(
            *output
                .shape
                .last()
                .ok_or("multimodal: empty output shape")?,
        );
    }
    let width = widths[0];
    if widths.iter().any(|&w| w != width) {
        return Err("multimodal: encoder output widths differ".into());
    }
    Ok(
        json!({"kind": "multimodal", "hidden": mm.hidden, "lm_hidden": width, "pad_token": mm.pad_token,
        "tokens": tokens, "slab_rows": mm.slab_rows, "table_capacity": mm.table_capacity,
        "slab_bytes": slab_bytes, "table_bytes": table_bytes}),
    )
}

/// The bundle's media obligation, or `None` when `main` has no speech or multimodal pipeline.
pub fn request(
    main: &MediaPacket<'_>,
    sidecars: &[MediaPacket<'_>],
) -> Result<Option<Value>, String> {
    let mut families = Vec::new();
    for p in main.pipelines {
        let family = match p.driver.as_str() {
            "causal.v1" if p.parameters.get("overlay_rows").is_some_and(|&r| r > 0) => {
                audio_lm(p, sidecars)?
            }
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

    fn pipeline(
        name: &str,
        driver: &str,
        params: &[(&str, u64)],
        strings: &[(&str, &str)],
        tensors: &[(&str, Vec<u64>)],
    ) -> PacketPipeline {
        PacketPipeline {
            name: name.into(),
            driver: driver.into(),
            programs: BTreeMap::from([("forward".into(), 0)]),
            tensors: tensors
                .iter()
                .map(|(role, shape)| {
                    (
                        role.to_string(),
                        PipelineTensor {
                            name: format!("t.{role}"),
                            dtype: PipelineDType::U32,
                            shape: shape.clone(),
                        },
                    )
                })
                .collect(),
            parameters: params.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            strings: strings
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn codec_lm() -> (Vec<PacketPipeline>, Vec<PacketPipeline>) {
        let lm = pipeline(
            "speech",
            "tts.codec_lm.v1",
            &[
                ("audio.sample_rate", 24000),
                ("codec.codebook", 4096),
                ("codec.frame_codes", 7),
                ("codec.frame_samples", 2048),
                ("audio.token_base", 128266),
                ("tokens.max_new_cap", 1400),
                ("prompt.prefix.count", 1),
                ("prompt.suffix.count", 3),
                ("max_context", 2048),
                ("stop.count", 1),
                ("stop.0", 128258),
            ],
            &[("codec.packet", "codec.pkt"), ("prompt.voice_token", "<v>")],
            &[],
        );
        let codec = pipeline(
            "codec.decode",
            "codec.v1",
            &[
                ("audio.sample_rate", 24000),
                ("codec.codebook", 4096),
                ("codec.frame_codes", 7),
                ("codec.frame_samples", 2048),
                ("stream.window_frames", 6),
                ("stream.lookahead_frames", 2),
            ],
            &[],
            &[("codes", vec![3584]), ("pcm", vec![1024, 1024])],
        );
        (vec![lm], vec![codec])
    }

    fn packet<'a>(file: &'a str, pipelines: &'a [PacketPipeline]) -> MediaPacket<'a> {
        MediaPacket {
            file,
            pipelines,
            vocabulary: None,
            multimodal: None,
            mm_tensors: None,
        }
    }

    #[test]
    fn codec_lm_request_reads_both_packets() {
        let (lm, codec) = codec_lm();
        let sidecars = [packet("model.pkt", &lm), packet("codec.pkt", &codec)];
        let request = request(&packet("model.pkt", &lm), &sidecars)
            .unwrap()
            .unwrap();
        let family = &request["families"][0];
        assert_eq!(family["kind"], "codec_lm");
        assert_eq!(family["prompt_tokens"], 5);
        assert_eq!(family["codes_capacity"], 3584);
        assert_eq!(family["pcm_capacity"], 1048576);
        assert_eq!(family["stops"], serde_json::json!([128258]));
    }

    #[test]
    fn stateful_codec_declares_its_steps() {
        let (lm, mut codec) = codec_lm();
        let windowed = request(&packet("model.pkt", &lm), &[packet("codec.pkt", &codec)])
            .unwrap()
            .unwrap();
        assert_eq!(windowed["families"][0]["stream_slots"], 0);
        assert_eq!(windowed["families"][0]["steps"], serde_json::json!([]));
        let c = &mut codec[0];
        for (k, v) in [
            ("stream.stateful", 1),
            ("stream.slots", 4),
            ("stream.step_frames", 17),
            ("lengths.count", 1),
        ] {
            c.parameters.insert(k.into(), v);
        }
        c.programs.insert("step.b1.f1".into(), 1);
        c.programs.insert("step.b4.f17".into(), 2);
        for (role, n) in [
            ("stream.slot", 4),
            ("stream.pos", 4),
            ("stream.rope_pos", 68),
            ("stream.n.0", 4),
        ] {
            c.tensors.insert(
                role.into(),
                PipelineTensor {
                    name: format!("in.codec.{role}"),
                    dtype: PipelineDType::U32,
                    shape: vec![n],
                },
            );
        }
        let family = &request(&packet("model.pkt", &lm), &[packet("codec.pkt", &codec)])
            .unwrap()
            .unwrap()["families"][0];
        assert_eq!(family["stream_slots"], 4);
        assert_eq!(family["steps"], serde_json::json!([[1, 1], [4, 17]]));
        assert_eq!(family["rope_capacity"], 68);
        assert_eq!(family["new_rows_capacities"], serde_json::json!([4]));
        codec[0].tensors.remove("stream.n.0");
        assert!(request(&packet("model.pkt", &lm), &[packet("codec.pkt", &codec)]).is_err());
    }

    #[test]
    fn clone_encoder_declares_its_capacities() {
        let (mut lm, codec) = codec_lm();
        lm[0]
            .strings
            .insert("clone.encoder".into(), "ref_encoder.pkt".into());
        lm[0].parameters.insert("clone.eos_code".into(), 0);
        let mut enc = pipeline(
            "codec.encode",
            "codec.encode.v1",
            &[
                ("audio.sample_rate", 24000),
                ("codec.frame_samples", 2048),
                ("codec.frame_codes", 7),
                ("codec.codebook", 4096),
            ],
            &[],
            &[("pcm", vec![131072]), ("codes", vec![64, 7])],
        );
        enc.programs = BTreeMap::from([("encode.f32".into(), 0), ("encode.f64".into(), 1)]);
        let enc = vec![enc];
        let sidecars = [packet("codec.pkt", &codec), packet("ref_encoder.pkt", &enc)];
        let family = &request(&packet("model.pkt", &lm), &sidecars)
            .unwrap()
            .unwrap()["families"][0];
        assert_eq!(family["clone_frames"], serde_json::json!([32, 64]));
        assert_eq!(family["clone_codes_capacity"], 448);
        assert_eq!(family["clone_frame_codes"], 7);
        assert!(
            request(&packet("model.pkt", &lm), &sidecars[..1]).is_err(),
            "declared encoder absent"
        );
    }

    #[test]
    fn missing_parameters_sidecars_and_overflow_reject() {
        let (mut lm, codec) = codec_lm();
        let sidecars = [packet("codec.pkt", &codec)];
        assert!(
            request(&packet("model.pkt", &lm), &[]).is_err(),
            "absent sidecar"
        );
        lm[0].parameters.remove("tokens.max_new_cap");
        assert!(
            request(&packet("model.pkt", &lm), &sidecars).is_err(),
            "missing parameter"
        );
        let text = vec![pipeline(
            "decode",
            "causal.v1",
            &[("overlay_rows", 0)],
            &[],
            &[],
        )];
        assert!(
            request(&packet("model.pkt", &text), &[]).unwrap().is_none(),
            "plain LM owes nothing"
        );
        let mut wide = codec.clone();
        wide[0].tensors.get_mut("pcm").unwrap().shape = vec![u64::MAX, 2];
        let (lm, _) = codec_lm();
        assert!(
            request(&packet("model.pkt", &lm), &[packet("codec.pkt", &wide)]).is_err(),
            "overflow"
        );
    }
}
