W = '/root/plow/.claude/worktrees/tts-veena-chatterbox/crates/'


def patch(path, pairs):
    s = open(W + path).read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, s.count(a), a[:70])
        s = s.replace(a, b)
    open(W + path, 'w').write(s)


patch('plow-asset/src/segment_roles.rs', [
    ("""pub fn cublaslt_prefill_bf16(profile: &str, m: u32, n: u32, k: u32) -> bool {""",
     """/// Llama-3.2-3B (Veena: hidden 3072, 24/8 heads, inter 8192) and Chatterbox T3 (Llama-520M:
/// hidden 1024, inter 4096) plain projections: q/o, k/v, down. The fused gate/up stays native.
/// On h200 the native object's 128-row segment cost ~1.95 ms per Veena layer (sm90a, 2026-09-26).
pub const CUBLASLT_PREFILL_LLAMA_TTS_SHAPES: [(u32, u32); 5] = [
    (3072, 3072),
    (1024, 3072),
    (3072, 8192),
    (1024, 1024),
    (1024, 4096),
];

pub fn cublaslt_prefill_bf16(profile: &str, m: u32, n: u32, k: u32) -> bool {"""),
    ("""        && (CUBLASLT_PREFILL_GEMMA4_SHAPES.contains(&(n, k))
            || CUBLASLT_PREFILL_GEMMA4_26B_SHAPES.contains(&(n, k)))""",
     """        && (CUBLASLT_PREFILL_GEMMA4_SHAPES.contains(&(n, k))
            || CUBLASLT_PREFILL_GEMMA4_26B_SHAPES.contains(&(n, k))
            || CUBLASLT_PREFILL_LLAMA_TTS_SHAPES.contains(&(n, k)))"""),
])
patch('devgen/src/lib.rs', [
    ("""        assert!(
            model_type.starts_with("gemma4")
                && arch == "sm_90a\"""", """        assert!(
            (model_type.starts_with("gemma4") || model_type == "llama")
                && arch == "sm_90a\""""),
    ("""            "cuBLASLt prefill emission requires Gemma 4 BF16 on single-GPU SM90\"""",
     """            "cuBLASLt prefill emission requires Gemma 4 or Llama BF16 on single-GPU SM90\""""),
])
print("ok")
