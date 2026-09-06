use super::*;

#[test]
fn packet_capabilities_are_explicit() {
    for model_type in ["gemma4", "gemma4_text", "llama", "qwen3"] {
        let capabilities = emit_capabilities(model_type);
        assert!(capabilities.dense_packet_contracts);
        assert!(capabilities.decode_objects);
        assert!(!capabilities.cublaslt_decode);
    }
    let qwen = emit_capabilities("qwen3_5");
    assert!(!qwen.dense_packet_contracts);
    assert!(qwen.decode_objects);
    assert!(qwen.cublaslt_decode);
    for model_type in ["kimi_k3", "glm5_next", "unknown"] {
        let capabilities = emit_capabilities(model_type);
        assert!(!capabilities.dense_packet_contracts);
        assert!(!capabilities.decode_objects);
        assert!(!capabilities.cublaslt_decode);
    }
}

#[test]
fn cublaslt_emission_rejects_unloadable_combinations() {
    let qwen = emit_capabilities("qwen3_5");
    assert!(cublaslt_emit_supported(qwen, "sm_90a", 1, false));
    assert!(!cublaslt_emit_supported(qwen, "sm_120", 1, false));
    assert!(!cublaslt_emit_supported(qwen, "gfx950", 1, false));
    assert!(!cublaslt_emit_supported(qwen, "sm_90a", 2, false));
    assert!(!cublaslt_emit_supported(qwen, "sm_90a", 1, true));
    assert!(!cublaslt_emit_supported(
        emit_capabilities("gemma4"),
        "sm_90a",
        1,
        false,
    ));
}
