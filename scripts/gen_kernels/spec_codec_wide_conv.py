"""gen_search block spec: the 12 Hz codec's residual-unit conv (causal Conv1dF32 k7, dilation 1/3/9,
96..768 channels) at the served step shapes. Generated bodies: codec_blocks.py (TileLang) and
codec_blocks_triton.py; current plow kernel: the emitted wgmma taps tile via codec_conv.cu.
"""
import codec_spec_common as cc

CLASSES = {
    "c96_b1": cc.conv_class("b1", 96, 1926, 1, 75.0),
    "c96_b8": cc.conv_class("b8", 96, 4081, 1, 1013.0),
    "c192_b8": cc.conv_class("b8", 192, 1362, 3, 522.0),
    "c384_b8": cc.conv_class("b8", 384, 341, 1, 364.0),
    "c768_b8": cc.conv_class("b8", 768, 69, 1, 350.0),
    "c768_b1": cc.conv_class("b1", 768, 38, 1, 179.0),
}

SPEC = {
    "name": "codec_wide_conv",
    "op": "Conv1dF32 causal k7 (12 Hz residual unit c1)",
    "dtype": "f32 (3xTF32; plow: split-bf16 wide / wgmma taps)",
    "classes": CLASSES,
    "cost": cc.conv_cost,
    "inputs": cc.conv_inputs,
    "rotate": ["x"],
    "out": lambda c: ((c["B"], c["T"], c["cout"]), "float32"),
    "reference": cc.conv_reference,
    # the served split-bf16 tiles are gated at 1e-4 against the golden (speech_f32_op_test)
    "tol": {"rel_l2": 1e-4},
    "generators": cc.conv_generators(
        {"bm": [128], "bn": [64, 96, 128], "bk": [32], "threads": [256], "stages": [2, 3]},
        {"prec": ["tf32x3", "tf32"], "bm": [64, 128], "bn": [64, 128], "bk": [32], "warps": [4, 8], "stages": [3]}),
    "current": {
        "label": "sp_wg_gemm taps tile as emitted (pinned, ragged, split-bf16, 64-wide, no output activation; flags 0x26e008), 132 blocks",
        "lib": cc.CURRENT_LIB,
        "bind": cc.conv_current_bind,
    },
}
