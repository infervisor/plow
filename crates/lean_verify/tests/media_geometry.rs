//! `media_geometry.v1`: the production families (values from the 4de53bf6 H100 bundles) are
//! accepted; each single-field mutation that breaks a contract is rejected.

use serde_json::{json, Value};

fn families() -> Vec<Value> {
    vec![
        json!({"kind": "audio_lm", "sample_rate": 16000, "max_seconds": 30, "encoder_sample_rate": 16000,
            "encoder_max_samples": 480000, "hop": 160, "chunk_frames": 100, "frame_stride": 8,
            "overlay_rows": 1500, "encoder_output_rows": 2496, "hidden": 2048, "encoder_width": 2048,
            "max_context": 2048, "max_tokens": 1024, "reserve_per_row": 1, "reserve_extra": 64,
            "audio_token": 151676, "stops": [151643, 151645]}),
        json!({"kind": "rnnt", "hop": 160, "max_samples": 479999, "input_frames": 3000,
            "transforms": [
                {"kernel": 3, "stride": 2, "pad_before": 2, "pad_after": 1},
                {"kernel": 3, "stride": 2, "pad_before": 2, "pad_after": 1},
                {"kernel": 1, "stride": 1, "pad_before": 0, "pad_after": 0},
                {"kernel": 3, "stride": 2, "pad_before": 2, "pad_after": 1},
                {"kernel": 1, "stride": 1, "pad_before": 0, "pad_after": 0}],
            "frames": 376, "joint_rows": 376, "blank": 13087, "vocab": 13087, "max_symbols_per_frame": 10}),
        json!({"kind": "codec_lm", "lm_sample_rate": 24000, "lm_codebook": 4096, "lm_frame_codes": 7,
            "lm_frame_samples": 2048, "token_base": 128266, "max_new_tokens": 1400, "prompt_tokens": 5,
            "max_context": 2048, "stops": [128258, 128262], "codec_sample_rate": 24000, "codebook": 4096,
            "frame_codes": 7, "frame_samples": 2048, "codes_capacity": 3584, "pcm_capacity": 1048576,
            "window_frames": 6, "lookahead_frames": 2}),
        json!({"kind": "guided_lm", "speech_vocab": 8194, "start_speech": 6561, "stop_speech": 6562,
            "valid_below": 6561, "text_vocab": 704, "start_text": 255, "stop_text": 0,
            "max_speech_tokens": 1000, "speech_positions": 4100, "overlay_rows": 512, "max_context": 2048}),
        json!({"kind": "vad", "frame_samples": 512, "context_samples": 64, "input_elements": 576,
            "state_banks": 2, "state_sizes": [128, 128, 128, 128], "min_speech_ms": 250,
            "max_duration_ms": 600000}),
        json!({"kind": "multimodal", "hidden": 3840, "lm_hidden": 3840, "pad_token": 0,
            "tokens": [258880, 255999, 256000], "slab_rows": 8192, "table_capacity": 16384,
            "slab_bytes": 62914560, "table_bytes": 131072}),
    ]
}

fn payload(families: Vec<Value>) -> Value {
    json!({"schema": 1, "families": families})
}

#[test]
#[ignore = "requires built plow_verify; CPU-only"]
fn production_media_contracts_hold_and_each_mutation_rejects() {
    let mut cases = vec![("production", payload(families()))];
    let mutations: Vec<(usize, &str, Value)> = vec![
        (0, "overlay_rows", json!(389)),
        (0, "encoder_width", json!(1024)),
        (0, "max_tokens", json!(1700)),
        (0, "stops", json!([151676])),
        (0, "frame_stride", json!(0)),
        (1, "frames", json!(375)),
        (1, "joint_rows", json!(300)),
        (1, "blank", json!(13088)),
        (1, "max_symbols_per_frame", json!(0)),
        (1, "input_frames", json!(2999)),
        (2, "codebook", json!(4095)),
        (2, "pcm_capacity", json!(1048575)),
        (2, "codes_capacity", json!(3585)),
        (2, "stops", json!([130000])),
        (2, "max_new_tokens", json!(2043)),
        (2, "window_frames", json!(-1)),
        (3, "start_speech", json!(8194)),
        (3, "speech_positions", json!(1001)),
        (4, "input_elements", json!(575)),
        (4, "state_sizes", json!([128, 128, 128])),
        (4, "kind", json!("vad2")),
        (5, "tokens", json!([2147483648u64])),
        (5, "table_capacity", json!(12288)),
        (5, "slab_bytes", json!(62914561)),
        (5, "lm_hidden", json!(2560)),
        (0, "unknown", json!(1)),
    ];
    for (family, field, value) in &mutations {
        let mut fs = families();
        fs[*family][*field] = value.clone();
        cases.push((field, payload(fs)));
    }
    cases.push(("empty", payload(vec![])));
    let requests: Vec<_> = cases.iter().map(|(_, p)| ("media_geometry.v1", p.clone())).collect();
    let certs = lean_verify::call_batch(&requests).unwrap();
    assert!(certs[0].ok, "{:?}", certs[0].reason);
    for ((what, _), cert) in cases.iter().zip(&certs).skip(1) {
        assert!(!cert.ok, "{what} mutation accepted");
    }
}
