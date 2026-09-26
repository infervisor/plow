"""Shared helpers for the S3Gen reference (PyTorch, fp32) used by s3gen_check.py."""
import glob
import os


def snapshot():
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return glob.glob(os.path.join(hf, "hub/models--ResembleAI--chatterbox/snapshots/*/"))[0]


def load_ref(device="cuda"):
    """Stock S3Token2Wav (fp32, eval) + the builtin voice's gen ref_dict on `device`."""
    import perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    import torch
    from safetensors.torch import load_file
    from chatterbox.models.s3gen import S3Gen
    m = S3Gen()
    m.load_state_dict(load_file(os.path.join(snapshot(), "s3gen.safetensors")), strict=False)
    m = m.to(device).eval()
    c = torch.load(os.path.join(snapshot(), "conds.pt"), map_location=device, weights_only=True)
    gen = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c["gen"].items()}
    return m, gen
