"""Catalog entry attn_pf_hd256_sliding (imported by build_catalog.py).

A hand-parameterized sm_90a wgmma template (runtime/nvidia/gen_attn_pf_hd256_sliding.cu) with
its own role wrapper, so the entry gives `build` / `tune` hooks instead of a TileLang body.
Standalone check / bench: attn_pf_hd256_sliding.py.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "attn_pf_hd256_sliding", Path(__file__).resolve().parent / "attn_pf_hd256_sliding.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ENTRIES = {
    "attn_pf_hd256_sliding": {
        # Any sliding window (crates/devgen/src/gen_kernels.rs ANY_SLIDING); the reference window
        # of build_catalog.py's own bench is the 12B/26B one.
        "signature": {"op": "flash_prefill", "head_dim": 256, "mask": "causal", "window": 1024,
                      "window_any": True, "gqa": [2, 16], "gqa_even": True, "ring_kv": True,
                      "dtype": "bf16", "kv_dtype": "bf16", "arch": "sm_90a"},
        "object": "gen_sm90a_attn_pf_hd256_sliding.cubin",
        "build": _mod.catalog_build,
        "tune": _mod.tune_row,
        "classes": [(h, kv, rows) for h, kv, w, rows in _mod.CLASSES if w == 1024],
    },
    # The FlashPrefillFp8 twin: e4m3 K/V with one f32 scale per (position, KV head) row
    # (crates/devgen/src/gen_kernels.rs KvDtype::Fp8). Same schedule; no tensor maps.
    "attn_pf_hd256_sliding_fp8kv": {
        "signature": {"op": "flash_prefill", "head_dim": 256, "mask": "causal", "window": 1024,
                      "window_any": True, "gqa": [2, 16], "gqa_even": True, "ring_kv": True,
                      "dtype": "bf16", "kv_dtype": "fp8_e4m3_rowscale", "arch": "sm_90a"},
        "object": "gen_sm90a_attn_pf_hd256_sliding_fp8kv.cubin",
        "build": lambda cfg, out: _mod.catalog_build(cfg, out, fp8=True),
        "tune": lambda: _mod.tune_row(fp8=True),
        "classes": [(h, kv, rows) for h, kv, w, rows in _mod.CLASSES if w == 1024],
    },
    # The FP8-KV twin for media-span sites (GEN_MEDIA_SPAN=1): rows of one bidirectional media
    # span also attend that span's later rows. A separate object keeps the causal one unchanged.
    "attn_pf_hd256_sliding_fp8kv_span": {
        "signature": {"op": "flash_prefill", "head_dim": 256, "mask": "causal_media_span",
                      "window": 1024, "window_any": True, "gqa": [2, 16], "gqa_even": True,
                      "ring_kv": True, "dtype": "bf16", "kv_dtype": "fp8_e4m3_rowscale",
                      "arch": "sm_90a"},
        "object": "gen_sm90a_attn_pf_hd256_sliding_fp8kv_span.cubin",
        "build": lambda cfg, out: _mod.catalog_build(cfg, out, fp8=True, span=True),
        "tune": lambda: _mod.tune_row(fp8=True, span=True),
        "classes": [(h, kv, rows) for h, kv, w, rows in _mod.CLASSES if w == 1024],
    },
}
