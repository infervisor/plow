use std::collections::{BTreeMap, BTreeSet};

pub const SECTION: &str = "segment_roles.json";
pub const INTERPRETER: u8 = 0;
pub const FP8_PREFILL_GEMM: u8 = 1;
pub const PREFILL_ATTENTION: u8 = 2;
pub const GEMV_CTA512: u8 = 3;
pub const FP8_M1: u8 = 4;
pub const CUBLASLT: u8 = 5;
pub const PREFILL_ATTENTION_HD512_WG32: u8 = 6;
pub const MXFP4_MOE: u8 = 7;
pub const NATIVE_DECODE_TC: u8 = 8;
pub const W8A16_PREFILL_M1: u8 = 9;
pub const PREFILL_ATTENTION_HD256_BKV64: u8 = 10;
pub const PREFILL_ATTENTION_HD256_BKV32: u8 = 11;
pub const BF16_PREFILL_GEMM_GLU_GEMMA4: u8 = 12;
pub const W8A8_PREFILL_GEMM_GLU_GEMMA4: u8 = 13;
pub const PREFILL_ATTENTION_HD256_GQA2_BKV32: u8 = 14;
pub const PREFILL_ATTENTION_HD512_PX4_BQ64: u8 = 15;
/// Library role, like [`CUBLASLT`]: one segment holding a layer's `MoeGroupGluGemmaPf` +
/// `MoeGroupDownGemmaPf` pair, which the CUDA runtime may serve with cuBLASLt grouped matmuls
/// (`PLOW_MOE_PF_LT`). Without the runtime knob the segment runs in the interpreter unchanged.
pub const MOE_PREFILL_CUBLASLT: u8 = 16;
/// Library role for a DECODE rung, like [`MOE_PREFILL_CUBLASLT`]: one segment holding a layer's
/// `MoeExpertGluNormGemma` + `MoeExpertDownGemma` pair on a rung that runs the grouped-MoE arm
/// (`PLOW_GEMMA_MOE_DEC_GROUP`), which the CUDA runtime may serve with cuBLASLt grouped matmuls
/// (`PLOW_MOE_DEC_LT`). Without the runtime knob the rung runs in the interpreter unchanged.
pub const MOE_DECODE_CUBLASLT: u8 = 17;
/// Generated-kernel catalog roles (`scripts/gen_kernels/build_catalog.py`, devgen
/// `gen_kernels.rs`): each catalog entry owns one ID in this range, and its object's ABI string
/// ([`GeneratedAbi`]) names the entry and its launch geometry.
pub const GENERATED_FIRST: u8 = 18;
pub const GENERATED_LAST: u8 = 25;
pub const MAX_ROLE: u8 = GENERATED_LAST;

pub fn is_generated(role: u8) -> bool {
    (GENERATED_FIRST..=GENERATED_LAST).contains(&role)
}

/// ABI of a generated flash-prefill role object: one persistent CTA per packet block, a
/// host-marshaled direct entry plus the packet entry, packed requests, successor counters.
pub const GENERATED_FLASH_PREFILL_ABI: &str = "gen_flash_prefill_v1";
/// The FP8-KV twin (`FlashPrefillFp8`: e4m3 K/V, one f32 scale per row): the same direct ABI
/// with the k/v scale vectors in the partial-output slots, and the packed request table taken
/// from the op's i[4] handle.
pub const GENERATED_FLASH_PREFILL_FP8KV_ABI: &str = "gen_flash_prefill_fp8kv_v1";

/// `<family>:<catalog entry>:block=<threads>:smem=<dynamic bytes>`; the grid is the packet grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedAbi {
    pub family: String,
    pub entry: String,
    pub block: u32,
    pub smem: u32,
}

impl GeneratedAbi {
    pub fn parse(abi: &str) -> Option<Self> {
        let mut parts = abi.split(':');
        let family = parts.next()?;
        let entry = parts.next()?;
        let block = parts.next()?.strip_prefix("block=")?.parse().ok()?;
        let smem = parts.next()?.strip_prefix("smem=")?.parse().ok()?;
        let valid_entry = !entry.is_empty()
            && entry
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
        (parts.next().is_none()
            && (family == GENERATED_FLASH_PREFILL_ABI
                || family == GENERATED_FLASH_PREFILL_FP8KV_ABI)
            && valid_entry
            && block > 0
            && block % 32 == 0
            && smem > 0)
            .then(|| Self { family: family.into(), entry: entry.into(), block, smem })
    }
    pub fn format(&self) -> String {
        format!("{}:{}:block={}:smem={}", self.family, self.entry, self.block, self.smem)
    }
    /// Whether the object reads an e4m3 KV cache (`FlashPrefillFp8`).
    pub fn fp8_kv(&self) -> bool {
        self.family == GENERATED_FLASH_PREFILL_FP8KV_ABI
    }
}

pub fn is_projection(role: u8) -> bool {
    matches!(role, CUBLASLT | NATIVE_DECODE_TC)
}

/// Measured Gemma-4-12B native decode plans, `[m, n, k, bk, splits]`: devgen plans routed
/// segments from this set, and the runtime falls back to it for an object without `decode_plan`.
pub const NATIVE_DECODE_BF16_SHAPES: [[u32; 5]; 54] = [
    [1, 512, 3840, 256, 8],
    [1, 2048, 3840, 256, 8],
    [1, 3840, 4096, 128, 4],
    [1, 3840, 8192, 128, 4],
    [1, 3840, 15360, 256, 4],
    [1, 4096, 3840, 256, 4],
    [1, 8192, 3840, 256, 1],
    [1, 15360, 3840, 128, 1],
    [1, 262144, 3840, 256, 1],
    [2, 512, 3840, 128, 8],
    [2, 2048, 3840, 128, 8],
    [2, 3840, 4096, 256, 4],
    [2, 3840, 8192, 256, 4],
    [2, 3840, 15360, 256, 4],
    [2, 4096, 3840, 256, 4],
    [2, 8192, 3840, 256, 1],
    [2, 15360, 3840, 128, 1],
    [2, 262144, 3840, 256, 1],
    [4, 512, 3840, 256, 8],
    [4, 2048, 3840, 256, 8],
    [4, 3840, 4096, 256, 4],
    [4, 3840, 8192, 128, 4],
    [4, 3840, 15360, 128, 4],
    [4, 4096, 3840, 128, 4],
    [4, 8192, 3840, 256, 1],
    [4, 15360, 3840, 256, 1],
    [4, 262144, 3840, 256, 1],
    [8, 512, 3840, 256, 8],
    [8, 2048, 3840, 128, 8],
    [8, 3840, 4096, 128, 4],
    [8, 3840, 8192, 256, 4],
    [8, 3840, 15360, 256, 4],
    [8, 4096, 3840, 256, 4],
    [8, 8192, 3840, 256, 1],
    [8, 15360, 3840, 128, 1],
    [8, 262144, 3840, 256, 1],
    [16, 512, 3840, 256, 8],
    [16, 2048, 3840, 256, 8],
    [16, 3840, 4096, 128, 4],
    [16, 3840, 8192, 128, 4],
    [16, 3840, 15360, 128, 4],
    [16, 4096, 3840, 256, 4],
    [16, 8192, 3840, 256, 4],
    [16, 15360, 3840, 128, 1],
    [16, 262144, 3840, 256, 1],
    [32, 512, 3840, 256, 8],
    [32, 2048, 3840, 256, 8],
    [32, 3840, 4096, 128, 4],
    [32, 3840, 8192, 128, 4],
    [32, 3840, 15360, 128, 4],
    [32, 4096, 3840, 128, 4],
    [32, 8192, 3840, 128, 1],
    [32, 15360, 3840, 128, 1],
    [32, 262144, 3840, 256, 1],
];

pub const CUBLASLT_PREFILL_MAX_ROWS: u32 = 16384;
/// Widest decode rung whose projections may run as cuBLASLt segments.
pub const CUBLASLT_DECODE_MAX_ROWS: u32 = 128;
pub const CUBLASLT_PREFILL_ROWS: [u32; 3] = [128, 256, 512];
/// 1088 / 1152 / 4160 are fine-grained rungs (`PLOW_PF_LADDER_APPEND`): BOS makes an N-token prompt
/// N+1 rows. Left out, such a rung ran every projection on the native GEMM object. Measured on
/// h100-sxm5 2026-09-21: 12B C1 TTFT at 1024 in 47.22 -> 46.82 ms on the 1088 rung; 1152 (26B and
/// 12B) and 4160 neutral.
/// 2176 is 2048+128, the same BOS-swallowing step as 1152 and 4224: without it a 2048-token
/// prompt has no qualified rung and every projection at that rung reverts to the native GEMM
/// object (Gemm segments 206 -> 121 in build.json).
pub const CUBLASLT_PREFILL_WIDE_ROWS: [u32; 13] = [
    1024, 1088, 1152, 2048, 2176, 4096, 4160, 4224, 8192, 8320, 12288, 12416, 16384,
];
/// Short-prompt speech rungs (`PLOW_PF_LADDER_APPEND`): codec-LM prompts are ~20-60 rows, guided
/// speech-LM prefills (voice rows + text) and audio-LM prefills (audio rows + prompt) ~150-420.
pub const CUBLASLT_PREFILL_SPEECH_ROWS: [u32; 2] = [64, 384];
pub const CUBLASLT_PREFILL_GEMMA4_SHAPES: [(u32, u32); 8] = [
    (15360, 3840),
    (2048, 3840),
    (3840, 15360),
    (4096, 3840),
    (3840, 4096),
    (8192, 3840),
    (512, 3840),
    (3840, 8192),
];

/// The same projection set for Gemma-4-26B-A4B (hidden 2816, dense inter 2112).
/// Kept as its own list rather than folded into the 12B one so each shape's
/// provenance stays readable; the two are disjoint (3840- vs 2816-keyed).
/// Sliding layers: q (4096), k/v (2048), o (2816 x 4096). Full layers: q (8192),
/// k (1024), o (2816 x 8192) — no v_proj, V is the raw k_proj (attention_k_eq_v).
/// Dense MLP: gate/up (2112), down (2816 x 2112). The 128-wide router and the
/// routed-expert GEMMs are NOT here: they are MoE ops, not dense projections.
pub const CUBLASLT_PREFILL_GEMMA4_26B_SHAPES: [(u32, u32); 8] = [
    (4096, 2816),
    (2048, 2816),
    (2816, 4096),
    (8192, 2816),
    (1024, 2816),
    (2816, 8192),
    (2112, 2816),
    (2816, 2112),
];

/// Gemma-4 E4B (hidden 2560, 8 q / 2 kv heads, sliding hd 256, full hd 512, inter 10240, 42
/// layers x 256 per-layer inputs): q/k/v/o for both layer kinds, unfused gate/up, down, the
/// per-layer input gate/projection and the per-layer model projection.
pub const CUBLASLT_PREFILL_GEMMA4_E4B_SHAPES: [(u32, u32); 11] = [
    (2048, 2560),
    (4096, 2560),
    (512, 2560),
    (1024, 2560),
    (2560, 2048),
    (2560, 4096),
    (10240, 2560),
    (2560, 10240),
    (256, 2560),
    (2560, 256),
    (10752, 2560),
];

/// Llama-3.2-3B (Veena: hidden 3072, 24/8 heads, inter 8192) and Chatterbox T3 (Llama-520M:
/// hidden 1024, inter 4096) projections: q/o, k/v, down, and gate/up when emitted unfused
/// (`PLOW_NO_GLU_FUSE`). On h200 the native object's 128-row segment cost ~1.95 ms per Veena
/// layer; q/k/v/o/down on cuBLASLt took TTFT@60 56.7 -> 45.3 ms (sm90a, 2026-09-26).
pub const CUBLASLT_PREFILL_LLAMA_TTS_SHAPES: [(u32, u32); 7] = [
    (3072, 3072),
    (1024, 3072),
    (3072, 8192),
    (8192, 3072),
    (1024, 1024),
    (1024, 4096),
    (4096, 1024),
];

/// Qwen3-1.7B decoder (Qwen3-ASR thinker: hidden 2048, 16/8 heads x 128, inter 6144): q/o, k/v,
/// unfused gate/up, down.
pub const CUBLASLT_PREFILL_QWEN3_1_7B_SHAPES: [(u32, u32); 4] = [(2048, 2048), (1024, 2048), (6144, 2048), (2048, 6144)];

pub fn cublaslt_prefill_fp8(profile: &str, m: u32, n: u32, k: u32) -> bool {
    matches!(profile, "sm90a" | "sm_90a")
        && [64, 128, 256, 512, 1024, 1088, 1152, 2048, 2112, 2176, 4096, 4160, 4224, 8192].contains(&m)
        && CUBLASLT_PREFILL_GEMMA4_SHAPES.contains(&(n, k))
        && !((n, k) == (3840, 15360) && (m == 1088 || m >= 2048))
        && !(m == 4160 && [(2048, 3840), (3840, 8192)].contains(&(n, k)))
        && !(m == 2112 && [(3840, 4096), (3840, 8192)].contains(&(n, k)))
}

pub fn cublaslt_prefill_bf16(profile: &str, m: u32, n: u32, k: u32) -> bool {
    // At M <= 512 the small set is the down projection (3840, 15360), the o projection
    // (3840, 8192) and the unfused gate/up (15360, 3840): measured on H100 2026-09-17, Lt
    // gate/up + a separate GeGLU pass beat the fused GLU role at every bucket Lt covered
    // (TTFT@1024 89.9 -> 56.4 ms), so the shape is admitted at the small rows too.
    // Every Gemma-4 projection at every rung: at 128 rows the native GEMM object still cost
    // ~50 us per launch for the q/k/v and sliding-o shapes (~176 launches per chunk) while the
    // cuBLASLt calls at the same M run 10-45 us (H100 2026-09-17, campaign tracker).
    matches!(profile, "sm90a" | "sm_90a")
        && (CUBLASLT_PREFILL_ROWS.contains(&m)
            || CUBLASLT_PREFILL_WIDE_ROWS.contains(&m)
            || CUBLASLT_PREFILL_SPEECH_ROWS.contains(&m))
        && (CUBLASLT_PREFILL_GEMMA4_SHAPES.contains(&(n, k))
            || CUBLASLT_PREFILL_GEMMA4_26B_SHAPES.contains(&(n, k))
            || CUBLASLT_PREFILL_GEMMA4_E4B_SHAPES.contains(&(n, k))
            || CUBLASLT_PREFILL_LLAMA_TTS_SHAPES.contains(&(n, k))
            || CUBLASLT_PREFILL_QWEN3_1_7B_SHAPES.contains(&(n, k)))
}

pub const PREFILL_ATTENTION_HD512_WG32_ABI: &str = "attention_sm90_hd512_wg32_v1";
pub const PREFILL_ATTENTION_HD512_PX4_BQ64_ABI: &str = "attention_sm90_hd512_px4_bq64_v5";
pub const PREFILL_ATTENTION_HD256_BKV64_ABI: &str = "attention_sm90_hd256_bkv64_v1";
pub const PREFILL_ATTENTION_HD256_BKV32_ABI: &str = "attention_sm90_hd256_bkv32_v1";
pub const PREFILL_ATTENTION_HD256_GQA2_BKV32_ABI: &str =
    "attention_sm90_hd256_gqa2_bkv32_v2";
pub const MXFP4_MOE_ABI: &str = "mxfp4_moe_sm90_v1";
pub const W8A16_PREFILL_M1_ABI: &str = "w8a16_prefill_m1_sm90_v1";
pub const BF16_PREFILL_GEMM_GLU_GEMMA4_ABI: &str = "gemm_glu_sm90_gemma4_4k8k_v1";
pub const W8A8_PREFILL_GEMM_GLU_GEMMA4_ABI: &str =
    "gemm_glu_w8a8_sm90_gemma4_4k8k_v2";

pub fn requires_object(role: u8) -> bool {
    matches!(
        role,
        FP8_PREFILL_GEMM
            | PREFILL_ATTENTION
            | GEMV_CTA512
            | FP8_M1
            | PREFILL_ATTENTION_HD512_WG32
            | MXFP4_MOE
            | NATIVE_DECODE_TC
            | W8A16_PREFILL_M1
            | PREFILL_ATTENTION_HD256_BKV64
            | PREFILL_ATTENTION_HD256_BKV32
            | BF16_PREFILL_GEMM_GLU_GEMMA4
            | W8A8_PREFILL_GEMM_GLU_GEMMA4
            | PREFILL_ATTENTION_HD256_GQA2_BKV32
            | PREFILL_ATTENTION_HD512_PX4_BQ64
    ) || is_generated(role)
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentRoles {
    pub version: u32,
    #[serde(deserialize_with = "unique_segment_objects")]
    pub objects: std::collections::BTreeMap<u8, SegmentObject>,
    pub programs: Vec<ProgramRoles>,
}
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentObject {
    pub abi: String,
    pub file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub promote_k512: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention: Option<AttentionCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gemm: Option<GemmCapability>,
    /// Native decode `[m, n, k, bk, splits]` per routed projection shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_plan: Option<Vec<[u32; 5]>>,
}
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionCapability {
    pub profile: String,
    pub dtype: String,
    pub head_dim: u32,
    pub query_tile: u32,
    pub kv_tile: u32,
    pub warps: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<AttentionShape>,
}
/// The model geometry a fixed-shape attention object is compiled for.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionShape {
    pub n_head: u32,
    pub n_kv_head: u32,
    /// Sliding window in rows; 0 is global attention.
    pub window: u32,
    pub arena_bytes: u32,
    /// Prefill rungs the object serves; absent means any rung.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<Vec<u32>>,
}
/// The shape and launch geometry a fixed-shape GEMM role object is compiled for.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GemmCapability {
    /// Prefill rungs the object serves, ascending; the object's min/max row globals are the ends.
    pub rows: Vec<u32>,
    pub n: u32,
    pub k: u32,
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub stages: u32,
    pub block: u32,
    pub arena_bytes: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile_band: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direct_entry: Option<u32>,
}

/// The Gemma-4 geometry the fixed-shape GEMM roles had before packets carried a descriptor;
/// the runtime holds descriptor-less packets to it.
pub fn legacy_gemm(role: u8) -> Option<GemmCapability> {
    let bf16 = GemmCapability {
        rows: vec![4096, 8192],
        n: 15360,
        k: 3840,
        bm: 128,
        bn: 128,
        bk: 64,
        stages: 4,
        block: 384,
        arena_bytes: 197696,
        tile_band: None,
        direct_entry: None,
    };
    match role {
        BF16_PREFILL_GEMM_GLU_GEMMA4 => Some(bf16),
        W8A8_PREFILL_GEMM_GLU_GEMMA4 => Some(GemmCapability {
            bk: 128,
            tile_band: Some(16),
            direct_entry: Some(1),
            ..bf16
        }),
        _ => None,
    }
}

/// Like [`legacy_gemm`], for the fixed-shape attention roles.
pub fn legacy_attention_shape(role: u8) -> Option<AttentionShape> {
    match role {
        PREFILL_ATTENTION_HD512_PX4_BQ64 => Some(AttentionShape {
            n_head: 16,
            n_kv_head: 1,
            window: 0,
            arena_bytes: 110592,
            rows: Some(vec![4096, 8192]),
        }),
        PREFILL_ATTENTION_HD256_GQA2_BKV32 => Some(AttentionShape {
            n_head: 16,
            n_kv_head: 8,
            window: 1024,
            arena_bytes: 141312,
            rows: None,
        }),
        _ => None,
    }
}

impl SegmentObject {
    /// The descriptor when the packet carries one, else [`legacy_gemm`].
    pub fn gemm_or_legacy(&self, role: u8) -> Option<GemmCapability> {
        self.gemm.clone().or_else(|| legacy_gemm(role))
    }
    /// The descriptor when the packet carries one, else [`legacy_attention_shape`].
    pub fn attention_shape_or_legacy(&self, role: u8) -> Option<AttentionShape> {
        self.attention
            .as_ref()
            .and_then(|a| a.shape.clone())
            .or_else(|| legacy_attention_shape(role))
    }
    /// The descriptor when the packet carries one, else [`NATIVE_DECODE_BF16_SHAPES`].
    pub fn decode_plan_or_legacy(&self) -> &[[u32; 5]] {
        self.decode_plan.as_deref().unwrap_or(&NATIVE_DECODE_BF16_SHAPES)
    }
}

fn ascending_rows(rows: &[u32], multiple: u32) -> bool {
    multiple > 0
        && !rows.is_empty()
        && rows.windows(2).all(|w| w[0] < w[1])
        && rows.iter().all(|&r| r > 0 && r % multiple == 0)
}

fn valid_gemm(role: u8, g: &GemmCapability) -> bool {
    matches!(role, BF16_PREFILL_GEMM_GLU_GEMMA4 | W8A8_PREFILL_GEMM_GLU_GEMMA4)
        && ascending_rows(&g.rows, g.bm)
        && [g.n, g.k, g.bn, g.bk, g.stages, g.arena_bytes].iter().all(|&v| v > 0)
        && g.n % g.bn == 0
        && g.k % g.bk == 0
        && g.block > 0
        && g.block % 128 == 0
        && if role == W8A8_PREFILL_GEMM_GLU_GEMMA4 {
            g.tile_band.is_some_and(|v| v > 0) && g.direct_entry == Some(1)
        } else {
            g.tile_band.is_none() && g.direct_entry.is_none()
        }
}

fn valid_attention_shape(role: u8, a: &AttentionCapability, s: &AttentionShape) -> bool {
    s.n_head > 0
        && s.n_kv_head > 0
        && s.n_head % s.n_kv_head == 0
        && s.arena_bytes > 0
        && s.rows.as_deref().is_none_or(|rows| ascending_rows(rows, a.query_tile))
        && match role {
            PREFILL_ATTENTION_HD512_PX4_BQ64 => s.window == 0 && s.rows.is_some(),
            PREFILL_ATTENTION_HD256_GQA2_BKV32 => s.window > 0 && s.n_head == 2 * s.n_kv_head,
            _ => false,
        }
}

fn valid_decode_plan(plan: &[[u32; 5]]) -> bool {
    let mut shapes = BTreeSet::new();
    !plan.is_empty()
        && plan.iter().all(|&[m, n, k, bk, splits]| {
            (1..=32).contains(&m)
                && n > 0
                && k > 0
                && matches!(bk, 128 | 256)
                && (1..=8).contains(&splits)
                && shapes.insert((m, n, k))
        })
}
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramRoles {
    pub index: usize,
    pub roles: Vec<u8>,
}

fn validate_generated(object: &SegmentObject) -> Result<(), String> {
    let abi = GeneratedAbi::parse(&object.abi);
    let valid_hash = object.sha256.as_deref().is_some_and(|s| {
        s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    });
    if !valid_hash
        || object.promote_k512.is_some()
        || object.file.is_empty()
        || std::path::Path::new(&object.file)
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
        || object.gemm.is_some()
        || object.decode_plan.is_some()
        || object.attention.as_ref().zip(abi.as_ref()).is_none_or(|(a, abi)| {
            a.shape.is_some()
                || a.profile != "sm90a"
                || a.dtype != "bf16"
                || a.head_dim == 0
                || a.query_tile == 0
                || a.kv_tile == 0
                || a.warps.checked_mul(32) != Some(abi.block)
        })
    {
        return Err("invalid generated packet segment object".into());
    }
    Ok(())
}

fn unique_segment_objects<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<BTreeMap<u8, SegmentObject>, D::Error> {
    struct Unique;
    impl<'de> serde::de::Visitor<'de> for Unique {
        type Value = BTreeMap<u8, SegmentObject>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("unique segment object IDs")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut a: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some((key, value)) = a.next_entry()? {
                if out.insert(key, value).is_some() {
                    return Err(serde::de::Error::custom("duplicate segment object ID"));
                }
            }
            Ok(out)
        }
    }
    d.deserialize_map(Unique)
}

impl SegmentRoles {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let value: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        value.validate_schema()?;
        Ok(value)
    }
    pub fn validate_schema(&self) -> Result<(), String> {
        if self.version != 1
            || self.programs.is_empty()
            || self.objects.keys().any(|&id| !requires_object(id))
        {
            return Err("unsupported packet segment roles".into());
        }
        for (&id, object) in &self.objects {
            if is_generated(id) {
                validate_generated(object)?;
                continue;
            }
            let abi = match id {
                FP8_PREFILL_GEMM => "fp8_gemm_tma128_v1",
                PREFILL_ATTENTION => "attention_sm90_hd256_v1",
                GEMV_CTA512 => "gemv_sm90_cta512_v1",
                FP8_M1 => crate::fp8_m1_role::ABI,
                PREFILL_ATTENTION_HD512_WG32 => PREFILL_ATTENTION_HD512_WG32_ABI,
                MXFP4_MOE => MXFP4_MOE_ABI,
                NATIVE_DECODE_TC => "gemv_transposed_sm90_bf16_v1",
                W8A16_PREFILL_M1 => W8A16_PREFILL_M1_ABI,
                PREFILL_ATTENTION_HD256_BKV64 => PREFILL_ATTENTION_HD256_BKV64_ABI,
                PREFILL_ATTENTION_HD256_BKV32 => PREFILL_ATTENTION_HD256_BKV32_ABI,
                BF16_PREFILL_GEMM_GLU_GEMMA4 => BF16_PREFILL_GEMM_GLU_GEMMA4_ABI,
                W8A8_PREFILL_GEMM_GLU_GEMMA4 => W8A8_PREFILL_GEMM_GLU_GEMMA4_ABI,
                PREFILL_ATTENTION_HD256_GQA2_BKV32 => {
                    PREFILL_ATTENTION_HD256_GQA2_BKV32_ABI
                }
                PREFILL_ATTENTION_HD512_PX4_BQ64 => PREFILL_ATTENTION_HD512_PX4_BQ64_ABI,
                _ => return Err("invalid packet segment object role".into()),
            };
            let valid_hash = |hash: Option<&str>| {
                hash.is_some_and(|s| {
                    s.len() == 64
                        && s.bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                })
            };
            let hd512_wg = AttentionCapability {
                profile: "sm90a".into(),
                dtype: "bf16".into(),
                head_dim: 512,
                query_tile: 64,
                kv_tile: 32,
                warps: 8,
                shape: None,
            };
            let hd512_px4 = AttentionCapability {
                profile: "sm90a".into(),
                dtype: "bf16".into(),
                head_dim: 512,
                query_tile: 32,
                kv_tile: 16,
                warps: 8,
                shape: None,
            };
            let hd512_wg16 = AttentionCapability {
                kv_tile: 16,
                ..hd512_wg.clone()
            };
            let hd512_wg64 = AttentionCapability {
                kv_tile: 64,
                ..hd512_wg.clone()
            };
            let hd512_px4_bq64 = AttentionCapability {
                kv_tile: 16,
                warps: 16,
                ..hd512_wg.clone()
            };
            let hd256_bkv64 = AttentionCapability {
                profile: "sm90a".into(),
                dtype: "bf16".into(),
                head_dim: 256,
                query_tile: 64,
                kv_tile: 64,
                warps: 8,
                shape: None,
            };
            let hd256_bkv32 = AttentionCapability {
                kv_tile: 32,
                ..hd256_bkv64.clone()
            };
            let unshaped = object
                .attention
                .as_ref()
                .map(|a| AttentionCapability { shape: None, ..a.clone() });
            if object.abi != abi
                || object.file.is_empty()
                || object.gemm.as_ref().is_some_and(|g| !valid_gemm(id, g))
                || object
                    .decode_plan
                    .as_deref()
                    .is_some_and(|plan| id != NATIVE_DECODE_TC || !valid_decode_plan(plan))
                || object.attention.as_ref().is_some_and(|a| {
                    a.shape.as_ref().is_some_and(|s| !valid_attention_shape(id, a, s))
                })
                || std::path::Path::new(&object.file)
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
                || (id == FP8_M1
                    && (object.promote_k512.is_none_or(|v| v > 1)
                        || !valid_hash(object.sha256.as_deref())
                        || object.attention.is_some()))
                || (id == PREFILL_ATTENTION_HD512_WG32
                    && (!valid_hash(object.sha256.as_deref())
                        || object.promote_k512.is_some()
                        || object
                            .attention
                            .as_ref()
                            .is_none_or(|a| {
                                a != &hd512_wg
                                    && a != &hd512_wg16
                                    && a != &hd512_wg64
                                    && a != &hd512_px4
                            })))
                || (id == PREFILL_ATTENTION_HD256_BKV64
                    && (!valid_hash(object.sha256.as_deref())
                        || object.promote_k512.is_some()
                        || object.attention.as_ref() != Some(&hd256_bkv64)))
                || (id == PREFILL_ATTENTION_HD256_BKV32
                    && (!valid_hash(object.sha256.as_deref())
                        || object.promote_k512.is_some()
                        || object.attention.as_ref() != Some(&hd256_bkv32)))
                || (id == PREFILL_ATTENTION_HD256_GQA2_BKV32
                    && (!valid_hash(object.sha256.as_deref())
                        || object.promote_k512.is_some()
                        || unshaped.as_ref() != Some(&hd256_bkv32)))
                || (id == PREFILL_ATTENTION_HD512_PX4_BQ64
                    && (!valid_hash(object.sha256.as_deref())
                        || object.promote_k512.is_some()
                        || unshaped.as_ref() != Some(&hd512_px4_bq64)))
                || (matches!(
                    id,
                    MXFP4_MOE
                        | NATIVE_DECODE_TC
                        | W8A16_PREFILL_M1
                        | BF16_PREFILL_GEMM_GLU_GEMMA4
                        | W8A8_PREFILL_GEMM_GLU_GEMMA4
                )
                    && (!valid_hash(object.sha256.as_deref())
                        || object.promote_k512.is_some()
                        || object.attention.is_some()))
                || (!matches!(
                    id,
                    FP8_M1
                        | PREFILL_ATTENTION_HD512_WG32
                        | MXFP4_MOE
                        | NATIVE_DECODE_TC
                        | W8A16_PREFILL_M1
                        | PREFILL_ATTENTION_HD256_BKV64
                        | PREFILL_ATTENTION_HD256_BKV32
                        | BF16_PREFILL_GEMM_GLU_GEMMA4
                        | W8A8_PREFILL_GEMM_GLU_GEMMA4
                        | PREFILL_ATTENTION_HD256_GQA2_BKV32
                        | PREFILL_ATTENTION_HD512_PX4_BQ64
                ) && (object.sha256.is_some()
                    || object.promote_k512.is_some()
                    || object.attention.is_some()))
            {
                return Err("invalid packet segment object".into());
            }
        }
        let mut programs = BTreeSet::new();
        let mut used = BTreeSet::new();
        for program in &self.programs {
            if !programs.insert(program.index)
                || program.roles.is_empty()
                || program.roles.iter().any(|&r| r > MAX_ROLE)
                || (program.roles.contains(&CUBLASLT) && program.roles.contains(&NATIVE_DECODE_TC))
            {
                return Err("invalid packet segment program".into());
            }
            used.extend(
                program
                    .roles
                    .iter()
                    .copied()
                    .filter(|&role| requires_object(role)),
            );
        }
        if used != self.objects.keys().copied().collect() {
            return Err("packet segment declarations do not match use".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fp8_padded_rungs_keep_native_losses_and_near_ties() {
        for m in [64, 1088, 1152, 2112, 2176, 4160, 4224] {
            for (n, k) in super::CUBLASLT_PREFILL_GEMMA4_SHAPES {
                let native = (n, k) == (3840, 15360) && (m == 1088 || m >= 2048)
                    || m == 4160 && [(2048, 3840), (3840, 8192)].contains(&(n, k))
                    || m == 2112 && [(3840, 4096), (3840, 8192)].contains(&(n, k));
                assert_eq!(super::cublaslt_prefill_fp8("sm90a", m, n, k), !native);
            }
        }
        assert!(!super::cublaslt_prefill_fp8("sm120", 1152, 4096, 3840));
        assert!(!super::cublaslt_prefill_fp8("sm90a", 1216, 4096, 3840));
    }

    use super::*;
    #[test]
    fn cublaslt_prefill_policy_is_exactly_the_measured_sm90_bf16_cells() {
        for profile in ["sm90a", "sm_90a"] {
            for m in CUBLASLT_PREFILL_ROWS {
                for (n, k) in CUBLASLT_PREFILL_GEMMA4_SHAPES {
                    assert!(cublaslt_prefill_bf16(profile, m, n, k));
                }
            }
            for m in CUBLASLT_PREFILL_WIDE_ROWS {
                for (n, k) in CUBLASLT_PREFILL_GEMMA4_SHAPES {
                    assert!(cublaslt_prefill_bf16(profile, m, n, k));
                }
            }
            for m in CUBLASLT_PREFILL_ROWS.iter().chain(&CUBLASLT_PREFILL_WIDE_ROWS) {
                for (n, k) in CUBLASLT_PREFILL_GEMMA4_26B_SHAPES {
                    assert!(cublaslt_prefill_bf16(profile, *m, n, k));
                }
            }
        }
        // The two model shape sets must stay disjoint, or a 12B admission silently
        // starts depending on a 26B entry (and the reverse) when either is edited.
        for shape in CUBLASLT_PREFILL_GEMMA4_26B_SHAPES {
            assert!(!CUBLASLT_PREFILL_GEMMA4_SHAPES.contains(&shape));
        }
        for (profile, m, n, k) in [
            ("sm120", 128, 3840, 15360),
            ("gfx942", 128, 3840, 8192),
            ("sm90a", 0, 3840, 15360),
            ("sm90a", 48, 3840, 15360),
            ("sm90a", 1024, 3840, 3840),
            ("sm90a", 1024, 15360, 8192),
            ("sm90a", 32768, 3840, 15360),
            ("sm90a", 128, 3840, 3840),
        ] {
            assert!(!cublaslt_prefill_bf16(profile, m, n, k));
        }
    }

    #[test]
    fn schema_rejects_duplicate_alias_ids_and_unknown_fields() {
        let object = r#"{"abi":"fp8_gemm_tma128_v1","file":"role.cubin"}"#;
        let raw = format!(
            r#"{{"version":1,"objects":{{"1":{object}}},"programs":[{{"index":0,"roles":[1]}}]}}"#
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for key in ["1", "01"] {
            let bad = raw.replace(
                &format!(r#""1":{object}"#),
                &format!(r#""1":{object},"{key}":{object}"#),
            );
            let error = SegmentRoles::from_bytes(bad.as_bytes()).unwrap_err();
            if key == "1" {
                assert!(error.contains("duplicate"), "{error}");
            }
        }
        for (needle, replacement) in [
            (r#""version":1"#, r#""version":1,"extra":0"#),
            (r#""abi":"#, r#""extra":0,"abi":"#),
            (r#""index":0"#, r#""index":0,"extra":0"#),
            (r#""version":1"#, r#""version":1,"version":1"#),
        ] {
            assert!(SegmentRoles::from_bytes(raw.replace(needle, replacement).as_bytes()).is_err());
        }
    }

    #[test]
    fn cublaslt_role_is_packet_only() {
        let raw = br#"{"version":1,"objects":{},"programs":[{"index":0,"roles":[0,5]}]}"#;
        SegmentRoles::from_bytes(raw).unwrap();
        for bad in [
            br#"{"version":1,"objects":{"5":{"abi":"x","file":"x"}},"programs":[{"index":0,"roles":[5]}]}"#.as_slice(),
            br#"{"version":1,"objects":{},"programs":[{"index":0,"roles":[6]}]}"#.as_slice(),
        ] {
            assert!(SegmentRoles::from_bytes(bad).is_err());
        }
    }

    #[test]
    fn hd512_attention_role_requires_exact_hash_and_capability() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"6":{{"abi":"attention_sm90_hd512_wg32_v1","file":"attention.cubin","sha256":"{}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":512,"query_tile":64,"kv_tile":32,"warps":8}}}}}},"programs":[{{"index":0,"roles":[0,6,6,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        SegmentRoles::from_bytes(raw.replace("\"kv_tile\":32", "\"kv_tile\":64").as_bytes())
            .unwrap();
        SegmentRoles::from_bytes(
            raw.replace(
                "\"query_tile\":64,\"kv_tile\":32",
                "\"query_tile\":32,\"kv_tile\":16",
            )
            .as_bytes(),
        )
        .unwrap();
        SegmentRoles::from_bytes(raw.replace("\"kv_tile\":32", "\"kv_tile\":16").as_bytes())
            .unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace("\"head_dim\":512", "\"head_dim\":256"),
            raw.replace("\"query_tile\":64", "\"query_tile\":32"),
            raw.replace("\"kv_tile\":32", "\"kv_tile\":128"),
            raw.replace("\"warps\":8", "\"warps\":4"),
            raw.replace("\"profile\":\"sm90a\"", "\"profile\":\"sm120\""),
            raw.replace("\"dtype\":\"bf16\"", "\"dtype\":\"fp8\""),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn hd512_px4_bq64_role_requires_512_thread_geometry() {
        let hash = "a".repeat(64);
        let raw = format!(
            r#"{{"version":1,"objects":{{"15":{{"abi":"{}","file":"attention.cubin","sha256":"{}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":512,"query_tile":64,"kv_tile":16,"warps":16}}}}}},"programs":[{{"index":0,"roles":[0,15,0]}}]}}"#,
            PREFILL_ATTENTION_HD512_PX4_BQ64_ABI, hash
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        assert!(SegmentRoles::from_bytes(raw.replace("\"warps\":16", "\"warps\":8").as_bytes())
            .is_err());
    }

    #[test]
    fn hd256_bkv64_attention_role_requires_exact_hash_and_capability() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"10":{{"abi":"attention_sm90_hd256_bkv64_v1","file":"attention.cubin","sha256":"{}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":256,"query_tile":64,"kv_tile":64,"warps":8}}}}}},"programs":[{{"index":0,"roles":[0,10,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace("\"head_dim\":256", "\"head_dim\":512"),
            raw.replace("\"query_tile\":64", "\"query_tile\":32"),
            raw.replace("\"kv_tile\":64", "\"kv_tile\":32"),
            raw.replace("\"warps\":8", "\"warps\":4"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn hd256_bkv32_attention_role_requires_exact_hash_and_capability() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"11":{{"abi":"attention_sm90_hd256_bkv32_v1","file":"attention.cubin","sha256":"{}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":256,"query_tile":64,"kv_tile":32,"warps":8}}}}}},"programs":[{{"index":0,"roles":[0,11,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace("\"head_dim\":256", "\"head_dim\":512"),
            raw.replace("\"query_tile\":64", "\"query_tile\":32"),
            raw.replace("\"kv_tile\":32", "\"kv_tile\":64"),
            raw.replace("\"warps\":8", "\"warps\":4"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn hd256_gqa2_bkv32_attention_role_requires_exact_hash_and_capability() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"14":{{"abi":"attention_sm90_hd256_gqa2_bkv32_v2","file":"attention.cubin","sha256":"{}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":256,"query_tile":64,"kv_tile":32,"warps":8}}}}}},"programs":[{{"index":0,"roles":[0,14,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace(
                PREFILL_ATTENTION_HD256_GQA2_BKV32_ABI,
                PREFILL_ATTENTION_HD256_BKV32_ABI,
            ),
            raw.replace("\"head_dim\":256", "\"head_dim\":512"),
            raw.replace("\"query_tile\":64", "\"query_tile\":32"),
            raw.replace("\"kv_tile\":32", "\"kv_tile\":64"),
            raw.replace("\"warps\":8", "\"warps\":4"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn generated_role_requires_abi_geometry_hash_and_capability() {
        let abi = GeneratedAbi {
            family: GENERATED_FLASH_PREFILL_ABI.into(),
            entry: "attn_pf_hd512".into(),
            block: 256,
            smem: 206848,
        };
        assert_eq!(GeneratedAbi::parse(&abi.format()), Some(abi.clone()));
        let raw = format!(
            r#"{{"version":1,"objects":{{"18":{{"abi":"{}","file":"gen.cubin","sha256":"{}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":512,"query_tile":64,"kv_tile":64,"warps":8}}}}}},"programs":[{{"index":0,"roles":[0,18,0]}}]}}"#,
            abi.format(),
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        let last = raw.replace("\"18\"", "\"25\"").replace("0,18,0", "0,25,0");
        SegmentRoles::from_bytes(last.as_bytes()).unwrap();
        let fp8 = raw.replace(GENERATED_FLASH_PREFILL_ABI, GENERATED_FLASH_PREFILL_FP8KV_ABI);
        SegmentRoles::from_bytes(fp8.as_bytes()).unwrap();
        let fp8_abi = GeneratedAbi {
            family: GENERATED_FLASH_PREFILL_FP8KV_ABI.into(),
            ..abi.clone()
        };
        assert!(GeneratedAbi::parse(&fp8_abi.format()).is_some_and(|abi| abi.fp8_kv()));
        assert!(!abi.fp8_kv());
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace("\"warps\":8", "\"warps\":4"),
            raw.replace("block=256", "block=250"),
            raw.replace(":smem=206848", ""),
            raw.replace("attn_pf_hd512", "Attn"),
            raw.replace(GENERATED_FLASH_PREFILL_ABI, "gen_flash_prefill_v0"),
            raw.replace("\"profile\":\"sm90a\"", "\"profile\":\"sm120\""),
            raw.replace("gen.cubin", "../gen.cubin"),
            raw.replace("\"18\"", "\"26\"").replace("0,18,0", "0,26,0"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err(), "{bad}");
        }
    }

    #[test]
    fn mxfp4_moe_role_requires_exact_hash() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"7":{{"abi":"mxfp4_moe_sm90_v1","file":"moe.cubin","sha256":"{}"}}}},"programs":[{{"index":0,"roles":[0,7,7,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&format!(r#","sha256":"{}""#, "a".repeat(64)), ""),
            raw.replace(&"a".repeat(64), "bad"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn native_decode_requires_hash_and_one_projection_backend() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"8":{{"abi":"gemv_transposed_sm90_bf16_v1","file":"native.cubin","sha256":"{}"}}}},"programs":[{{"index":0,"roles":[0,8,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace("[0,8,0]", "[5,8,0]"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn native_w8a16_m1_requires_exact_abi_and_hash() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"9":{{"abi":"w8a16_prefill_m1_sm90_v1","file":"m1.cubin","sha256":"{}"}}}},"programs":[{{"index":0,"roles":[0,9,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace("w8a16_prefill_m1_sm90_v1", "w8a16_prefill_small_sm90_v1"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    #[test]
    fn gemma4_bf16_gemm_glu_requires_exact_abi_and_hash() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"12":{{"abi":"gemm_glu_sm90_gemma4_4k8k_v1","file":"glu.cubin","sha256":"{}"}}}},"programs":[{{"index":0,"roles":[0,12,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace(
                BF16_PREFILL_GEMM_GLU_GEMMA4_ABI,
                "gemm_glu_sm90_gemma4_v0",
            ),
            raw.replace("glu.cubin", "../glu.cubin"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }

    fn object_raw(id: u8, object: &str) -> String {
        format!(
            r#"{{"version":1,"objects":{{"{id}":{object}}},"programs":[{{"index":0,"roles":[0,{id},0]}}]}}"#
        )
    }

    #[test]
    fn gemm_descriptor_round_trips_and_is_validated() {
        let hash = "a".repeat(64);
        for (id, abi) in [
            (BF16_PREFILL_GEMM_GLU_GEMMA4, BF16_PREFILL_GEMM_GLU_GEMMA4_ABI),
            (W8A8_PREFILL_GEMM_GLU_GEMMA4, W8A8_PREFILL_GEMM_GLU_GEMMA4_ABI),
        ] {
            let legacy = object_raw(id, &format!(r#"{{"abi":"{abi}","file":"glu.cubin","sha256":"{hash}"}}"#));
            let parsed = SegmentRoles::from_bytes(legacy.as_bytes()).unwrap();
            assert!(parsed.objects[&id].gemm.is_none());
            assert_eq!(parsed.objects[&id].gemm_or_legacy(id), legacy_gemm(id));
            assert_eq!(serde_json::to_string(&parsed).unwrap(), legacy);

            let mut roles = parsed;
            roles.objects.get_mut(&id).unwrap().gemm = legacy_gemm(id);
            let raw = serde_json::to_string(&roles).unwrap();
            let back = SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
            assert_eq!(back.objects[&id].gemm, legacy_gemm(id));
            let other = GemmCapability { rows: vec![2048, 4096], n: 2112 * 2, k: 2816, ..legacy_gemm(id).unwrap() };
            roles.objects.get_mut(&id).unwrap().gemm = Some(other.clone());
            roles.validate_schema().unwrap();
            assert_eq!(roles.objects[&id].gemm_or_legacy(id), Some(other.clone()));
            for bad in [
                GemmCapability { rows: vec![], ..other.clone() },
                GemmCapability { rows: vec![4096, 2048], ..other.clone() },
                GemmCapability { rows: vec![100], ..other.clone() },
                GemmCapability { k: 2817, ..other.clone() },
                GemmCapability { n: 0, ..other.clone() },
                GemmCapability { block: 100, ..other.clone() },
                GemmCapability {
                    tile_band: if id == W8A8_PREFILL_GEMM_GLU_GEMMA4 { None } else { Some(16) },
                    ..other.clone()
                },
            ] {
                roles.objects.get_mut(&id).unwrap().gemm = Some(bad);
                assert!(roles.validate_schema().is_err());
            }
        }
        let attention = object_raw(
            11,
            &format!(r#"{{"abi":"{PREFILL_ATTENTION_HD256_BKV32_ABI}","file":"a.cubin","sha256":"{hash}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":256,"query_tile":64,"kv_tile":32,"warps":8}},"gemm":{{"rows":[4096],"n":128,"k":128,"bm":128,"bn":128,"bk":64,"stages":4,"block":384,"arena_bytes":1}}}}"#),
        );
        assert!(SegmentRoles::from_bytes(attention.as_bytes()).is_err());
    }

    #[test]
    fn attention_shape_round_trips_and_is_validated() {
        let hash = "a".repeat(64);
        let gqa2 = object_raw(
            PREFILL_ATTENTION_HD256_GQA2_BKV32,
            &format!(r#"{{"abi":"{PREFILL_ATTENTION_HD256_GQA2_BKV32_ABI}","file":"a.cubin","sha256":"{hash}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":256,"query_tile":64,"kv_tile":32,"warps":8,"shape":{{"n_head":32,"n_kv_head":16,"window":512,"arena_bytes":141312}}}}}}"#),
        );
        let parsed = SegmentRoles::from_bytes(gqa2.as_bytes()).unwrap();
        assert_eq!(serde_json::to_string(&parsed).unwrap(), gqa2);
        let shape = parsed.objects[&PREFILL_ATTENTION_HD256_GQA2_BKV32]
            .attention_shape_or_legacy(PREFILL_ATTENTION_HD256_GQA2_BKV32)
            .unwrap();
        assert_eq!((shape.n_head, shape.n_kv_head, shape.window, shape.rows), (32, 16, 512, None));
        for bad in [
            gqa2.replace(r#""n_head":32"#, r#""n_head":48"#),
            gqa2.replace(r#""window":512"#, r#""window":0"#),
            gqa2.replace(r#""arena_bytes":141312"#, r#""arena_bytes":0"#),
            gqa2.replace(r#""window":512"#, r#""window":512,"rows":[100]"#),
            gqa2.replace(r#""window":512"#, r#""window":512,"extra":1"#),
            gqa2.replace(PREFILL_ATTENTION_HD256_GQA2_BKV32_ABI, PREFILL_ATTENTION_HD256_BKV32_ABI)
                .replace(r#""14""#, r#""11""#)
                .replace("0,14,0", "0,11,0"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err(), "{bad}");
        }
        let px4 = object_raw(
            PREFILL_ATTENTION_HD512_PX4_BQ64,
            &format!(r#"{{"abi":"{PREFILL_ATTENTION_HD512_PX4_BQ64_ABI}","file":"a.cubin","sha256":"{hash}","attention":{{"profile":"sm90a","dtype":"bf16","head_dim":512,"query_tile":64,"kv_tile":16,"warps":16}}}}"#),
        );
        let legacy = SegmentRoles::from_bytes(px4.as_bytes()).unwrap();
        assert_eq!(
            legacy.objects[&PREFILL_ATTENTION_HD512_PX4_BQ64]
                .attention_shape_or_legacy(PREFILL_ATTENTION_HD512_PX4_BQ64),
            legacy_attention_shape(PREFILL_ATTENTION_HD512_PX4_BQ64)
        );
        let shaped = px4.replace(
            r#""warps":16"#,
            r#""warps":16,"shape":{"n_head":8,"n_kv_head":1,"window":0,"arena_bytes":110592,"rows":[2048,4096]}"#,
        );
        SegmentRoles::from_bytes(shaped.as_bytes()).unwrap();
        for bad in [
            shaped.replace(r#","rows":[2048,4096]"#, ""),
            shaped.replace(r#""window":0"#, r#""window":1024"#),
            shaped.replace("[2048,4096]", "[4096,2048]"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err(), "{bad}");
        }
    }

    #[test]
    fn decode_plan_round_trips_and_is_validated() {
        let raw = object_raw(
            NATIVE_DECODE_TC,
            &format!(r#"{{"abi":"gemv_transposed_sm90_bf16_v1","file":"n.cubin","sha256":"{}","decode_plan":[[1,512,3840,256,8],[8,2048,2048,128,1]]}}"#, "a".repeat(64)),
        );
        let parsed = SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        assert_eq!(serde_json::to_string(&parsed).unwrap(), raw);
        assert_eq!(
            parsed.objects[&NATIVE_DECODE_TC].decode_plan_or_legacy(),
            [[1, 512, 3840, 256, 8], [8, 2048, 2048, 128, 1]]
        );
        let legacy = raw.replace(r#","decode_plan":[[1,512,3840,256,8],[8,2048,2048,128,1]]"#, "");
        assert_eq!(
            SegmentRoles::from_bytes(legacy.as_bytes()).unwrap().objects[&NATIVE_DECODE_TC]
                .decode_plan_or_legacy(),
            NATIVE_DECODE_BF16_SHAPES
        );
        for bad in [
            raw.replace("[8,2048,2048,128,1]", "[1,512,3840,128,8]"),
            raw.replace("[8,2048,2048,128,1]", "[64,2048,2048,128,1]"),
            raw.replace("[8,2048,2048,128,1]", "[8,2048,2048,64,1]"),
            raw.replace("[8,2048,2048,128,1]", "[8,2048,2048,128,16]"),
            raw.replace(r#"[[1,512,3840,256,8],[8,2048,2048,128,1]]"#, "[]"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err(), "{bad}");
        }
        let misplaced = object_raw(
            W8A16_PREFILL_M1,
            &format!(r#"{{"abi":"{W8A16_PREFILL_M1_ABI}","file":"m.cubin","sha256":"{}","decode_plan":[[1,512,3840,256,8]]}}"#, "a".repeat(64)),
        );
        assert!(SegmentRoles::from_bytes(misplaced.as_bytes()).is_err());
    }

    #[test]
    fn gemma4_w8a8_gemm_glu_requires_exact_abi_and_hash() {
        let raw = format!(
            r#"{{"version":1,"objects":{{"13":{{"abi":"gemm_glu_w8a8_sm90_gemma4_4k8k_v2","file":"glu.cubin","sha256":"{}"}}}},"programs":[{{"index":0,"roles":[0,13,0]}}]}}"#,
            "a".repeat(64)
        );
        SegmentRoles::from_bytes(raw.as_bytes()).unwrap();
        for bad in [
            raw.replace(&"a".repeat(64), "bad"),
            raw.replace(
                W8A8_PREFILL_GEMM_GLU_GEMMA4_ABI,
                "gemm_glu_w8a8_sm90_gemma4_v0",
            ),
            raw.replace("glu.cubin", "../glu.cubin"),
        ] {
            assert!(SegmentRoles::from_bytes(bad.as_bytes()).is_err());
        }
    }
}
