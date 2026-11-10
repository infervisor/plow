"""gen_search block spec: SnakeBeta + causal pad + Conv1dF32 k7 fused (the 12 Hz residual unit:
input SnakeBeta over the stream-tail window, conv, output SnakeBeta) at the served b8 shapes.
Current plow: UnaryF32 kind 19 + conv (+ UnaryF32), recorded per unit.
"""
import codec_spec_common as cc

CLASSES = {
    "c96_b8_in": cc.conv_class("b8", 96, 4081, 1, 178.0 + 1013.0, pre=True),
    "c96_b8_inout": cc.conv_class("b8", 96, 4081, 1, 2 * 178.0 + 1013.0, pre=True, post=True),
    "c192_b8_in": cc.conv_class("b8", 192, 1362, 3, 120.0 + 522.0, pre=True),
    "c192_b8_inout": cc.conv_class("b8", 192, 1362, 3, 2 * 120.0 + 522.0, pre=True, post=True),
    "c384_b8_inout": cc.conv_class("b8", 384, 341, 1, 2 * 55.0 + 364.0, pre=True, post=True),
}

SPEC = {
    "name": "codec_snake_conv",
    "op": "SnakeBeta + causal pad + Conv1dF32 k7 (+ SnakeBeta)",
    "dtype": "f32 (3xTF32 conv, f32 snake)",
    "classes": CLASSES,
    "cost": cc.conv_cost,
    "inputs": cc.conv_inputs,
    "rotate": ["x"],
    "out": lambda c: ((c["B"], c["T"], c["cout"]), "float32"),
    "reference": cc.conv_reference,
    "tol": {"rel_l2": 1e-5},
    "generators": cc.conv_generators(
        {"bm": [128], "bn": [96, 128], "bk": [32], "threads": [256], "stages": [2]},
        {"bm": [128], "bn": [64, 128], "bk": [32], "warps": [4], "stages": [3]}),
    "current": {
        "label": "UnaryF32 SnakeBeta + conv tile (+ UnaryF32), packet_bench --sweep per unit",
        "measured_us": {n: c["us"] for n, c in CLASSES.items()},
        "source": "tts-perf/tp-codec/verify/l7a (qwen3-tts codec step, tp-codec 14759885b)",
    },
}
