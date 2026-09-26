W = '/root/plow/.claude/worktrees/tts-veena-chatterbox/'


def patch(path, pairs):
    s = open(W + path).read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, s.count(a), a[:70])
        s = s.replace(a, b)
    open(W + path, 'w').write(s)


s = open(W + 'crates/devgen/src/config.rs').read()
s = s.replace("""    pub(crate) speech_pos_rows: u32,
""", """    pub(crate) speech_pos_rows: u32,
    // `chatterbox_t3` contract scalars for the speech pipeline (floats as `<key>_f32` bit patterns).
    pub(crate) speech_params: Vec<(String, u64)>,
""", 1)
assert s.count("        speech_pos_rows: 0,\n") == 5
s = s.replace("        speech_pos_rows: 0,\n", "        speech_pos_rows: 0,\n        speech_params: Vec::new(),\n")
old = """        assert!(c.encoder_overlay_rows > 0 && c.speech_pos_rows > 0);
    }"""
new = """        assert!(c.encoder_overlay_rows > 0 && c.speech_pos_rows > 0);
        for (k, v) in t3.as_object().expect("chatterbox_t3 object") {
            if let Some(n) = v.as_u64() {
                c.speech_params.push((k.clone(), n));
            } else if let Some(f) = v.as_f64() {
                c.speech_params.push((format!("{k}_f32"), u64::from((f as f32).to_bits())));
            }
        }
    }"""
assert s.count(old) == 1
s = s.replace(old, new)
open(W + 'crates/devgen/src/config.rs', 'w').write(s)

patch('crates/devgen/src/lib.rs', [
    ("    if c.encoder_overlay_rows > 0 && !block_mode {\n        sections.push(\n            pipeline::causal_pipeline_section(",
     """    if c.speech_pos_rows > 0 && !block_mode {
        sections.push(
            tts::t3_pipeline_section(
                &m,
                pipeline::CausalPipelineSpec {
                    name: "speech",
                    max_context: ctx,
                    hidden: c.hidden,
                    decode_capacity: dbatch,
                    overlay_rows: c.encoder_overlay_rows,
                    ordered_dispatch: false,
                    tensors: pipeline::CausalPipelineTensors {
                        tokens: emitter.tn.ids,
                        positions: emitter.tn.pos,
                        kv_lengths: emitter.tn.kvlen,
                        overlay: Some(emitter.tn.encoder_overlay),
                        overlay_index: Some(emitter.tn.encoder_overlay_index),
                    },
                },
                emitter.tn.pos_base,
                &c.speech_params,
            )
            .unwrap_or_else(|error| panic!("T3 speech packet pipeline: {error}")),
        );
    } else if c.encoder_overlay_rows > 0 && !block_mode {
        sections.push(
            pipeline::causal_pipeline_section("""),
    ("    let audio_blob = (c.encoder_overlay_rows > 0 && !block_mode).then(|| {",
     "    let audio_blob = (c.encoder_overlay_rows > 0 && c.speech_pos_rows == 0 && !block_mode).then(|| {"),
])

patch('crates/devgen/src/tts.rs', [
    ("""pub fn profile(name: &str)""", """/// Chatterbox T3 (`tts.t3_cfg.v1`): a causal LM whose prefill rows are host embeddings (the
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
    meta.validate(model.progs.len(), |name| model.tensors.iter().find(|t| t.name == name).map(|t| t.bytes))?;
    Ok(SectionData { kind: causal.kind, name: SECTION.into(), data: serde_json::to_vec(&meta).map_err(|e| e.to_string())? })
}

pub fn profile(name: &str)"""),
])
print("ok")
