//! `media_geometry.v1`: the production families (values from the 4de53bf6 H100 bundles; the
//! stateful codec LM from the Fish S2 Pro stream-step bundle) are accepted; each single-field mutation that breaks a contract is rejected.

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
            "lm_frame_samples": 2048, "token_base": 128266, "audio_tokens": 28672, "max_new_tokens": 1400, "prompt_tokens": 5,
            "max_context": 2048, "stops": [128258, 128262], "codec_sample_rate": 24000, "codebook": 4096,
            "frame_codes": 7, "frame_samples": 2048, "codes_capacity": 3584, "pcm_capacity": 1048576,
            "window_frames": 6, "lookahead_frames": 2, "stream_slots": 0, "step_frames": 0, "steps": [],
            "slot_capacity": 0, "pos_capacity": 0, "rope_capacity": 0, "new_rows_capacities": []}),
        json!({"kind": "codec_lm", "lm_sample_rate": 44100, "lm_codebook": 4096, "lm_frame_codes": 10,
            "lm_frame_samples": 2048, "token_base": 151678, "audio_tokens": 4096, "max_new_tokens": 1024,
            "prompt_tokens": 21, "max_context": 4096, "stops": [151645], "codec_sample_rate": 44100,
            "codebook": 4096, "frame_codes": 10, "frame_samples": 2048, "codes_capacity": 10240,
            "pcm_capacity": 2097152, "window_frames": 48, "lookahead_frames": 0, "stream_slots": 32,
            "step_frames": 17, "steps": [[1, 1], [1, 17], [1, 4], [16, 17], [32, 17], [4, 17], [8, 17]],
            "slot_capacity": 32, "pos_capacity": 32, "rope_capacity": 544,
            "new_rows_capacities": [32, 32, 32, 32, 32, 32, 32]}),
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
        (4, "start_speech", json!(8194)),
        (4, "speech_positions", json!(1001)),
        (5, "input_elements", json!(575)),
        (5, "state_sizes", json!([128, 128, 128])),
        (5, "kind", json!("vad2")),
        (6, "tokens", json!([2147483648u64])),
        (6, "table_capacity", json!(12288)),
        (6, "slab_bytes", json!(62914561)),
        (6, "lm_hidden", json!(2560)),
        (3, "steps", json!([[33, 17]])),
        (3, "steps", json!([[1, 18]])),
        (3, "steps", json!([[1, 0]])),
        (3, "step_frames", json!(16)),
        (3, "rope_capacity", json!(543)),
        (3, "pos_capacity", json!(31)),
        (3, "slot_capacity", json!(31)),
        (
            3,
            "new_rows_capacities",
            json!([32, 32, 32, 31, 32, 32, 32]),
        ),
        (3, "stream_slots", json!(0)),
        (3, "codes_capacity", json!(5430)),
        (2, "steps", json!([[1, 1]])),
        (0, "unknown", json!(1)),
    ];
    for (family, field, value) in &mutations {
        let mut fs = families();
        fs[*family][*field] = value.clone();
        cases.push((field, payload(fs)));
    }
    cases.push(("empty", payload(vec![])));
    let requests: Vec<_> = cases
        .iter()
        .map(|(_, p)| ("media_geometry.v1", p.clone()))
        .collect();
    let certs = lean_verify::call_batch(&requests).unwrap();
    assert!(certs[0].ok, "{:?}", certs[0].reason);
    for ((what, _), cert) in cases.iter().zip(&certs).skip(1) {
        assert!(!cert.ok, "{what} mutation accepted");
    }
}
