"""Catalog entries for generated flash-decode roles (imported by build_catalog.py).

The TileLang body is attn_decode.py (one query row per sequence, GQA group per item, slot table,
split-KV with an in-kernel merge); the wrapper is runtime/nvidia/gen_flash_decode.cu. Standalone
bench: bench_decode.py. A class is (heads, kv heads, rows, context).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "attn_decode", Path(__file__).resolve().parent / "attn_decode.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

NSPLIT_MAX = _mod.NSPLIT_MAX

ENTRIES = {
    # Qwen3.5-class full-attention layers (24 / 4 heads, hd256): the GQA ratio is compiled in,
    # window and KV ring are launch operands. plow FlashDecode at B16 x 8K runs 606 us per
    # layer-call (3.8x the HBM floor); this body 188 us (1.2x).
    "attn_dec_hd256_gqa6": {
        "signature": {"op": "flash_decode", "head_dim": 256, "gqa": [6, 6], "window": 0,
                      "window_any": True, "ring_kv": True, "nsplit_max": NSPLIT_MAX,
                      "dtype": "bf16", "kv_dtype": "bf16", "arch": "sm_90a"},
        "object": "gen_sm90a_attn_dec_hd256_gqa6.cubin",
        # Unchecked global indexing (the launch is in bounds by construction; attn_decode.py's
        # standalone bench measures the same body).
        "pass_configs": ("TL_DISABLE_SAFE_MEMORY_ACCESS",),
        # bf16 output vs an fp32 reference: ~2.2e-3 (the prefill entry's bound).
        "tol": 4e-3,
        "body": lambda c: _mod.build(256, 6, c["bn"], c["stages"], c["threads"]),
        "sweep": [{"bn": bn, "stages": st, "threads": thr}
                  for bn, st, thr in [(64, 2, 128), (64, 3, 128), (32, 2, 128), (32, 3, 128)]],
        "classes": [(24, 4, 8, 4096), (24, 4, 16, 8192), (24, 4, 16, 15000), (24, 4, 48, 4096),
                    (24, 4, 1, 15000)],
    },
    # The gqa6 body over head-interleaved KV rows (PLOW_QWEN_KV_INTERLEAVE); the gqa6 config.
    "attn_dec_hd256_gqa6_ilv": {
        "signature": {"op": "flash_decode", "head_dim": 256, "gqa": [6, 6], "window": 0,
                      "window_any": True, "ring_kv": True, "nsplit_max": NSPLIT_MAX,
                      "dtype": "bf16", "kv_dtype": "bf16", "kv_layout": "row_interleaved",
                      "arch": "sm_90a"},
        "object": "gen_sm90a_attn_dec_hd256_gqa6_ilv.cubin",
        "pass_configs": ("TL_DISABLE_SAFE_MEMORY_ACCESS",),
        "tol": 4e-3,
        "defines": ("-DPLOW_GEN_KV_ILV=1",),
        "body": lambda c: _mod.build(256, 6, c["bn"], c["stages"], c["threads"], ilv=True),
        "classes": [(24, 4, 16, 8192), (24, 4, 48, 4096)],
    },
}
