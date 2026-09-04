#!/usr/bin/env python3
"""Drop-in replacement for `vllm serve` / `python -m vllm ...` that patches
three independent bugs blocking vLLM 0.27.0 from serving `google/gemma-4-12B-it`
at all. See `perf-data/gemma4-12b-sandbox-5090-2026-08-25.md` §3 for the full
writeup and `perf-data/prefill-occupancy-handoff-2026-08-25.md` for how this is
used in the live plow-vs-vLLM comparison harness.

Usage: `python3 perf-data/tools/vllm_gemma4_launch.py serve <model-dir> [vllm serve flags...]`
(same CLI as `vllm serve` — this only wraps import-time patching, then hands
off to vLLM's real entrypoint unmodified.)

Launcher for `vllm serve` that patches a transformers strictness check.

vLLM 0.27.0's ModelConfig reads `hf_text_config.head_dim` as a global
convenience accessor before any `--hf-overrides` reach the nested text_config
object, so heterogeneous (Gemma-4 sliding/full attention) configs raise
AmbiguousGlobalPerLayerAttributeError before the override can apply. This
patches the default of `allow_global_per_layer_attribute_access` to True at
the transformers class level, in-process, without touching the installed
package on disk.
"""
import sys
import types

# Stub out flashinfer.comm.fd_exchange: its `array.array[int]` return-type
# annotation is evaluated eagerly at import time and only valid on Python
# 3.13+ (array.array gained __class_getitem__ there). Under 3.11 this raises
# `TypeError: type 'array.array' is not subscriptable` as soon as vLLM's
# compilation/fusion pass machinery imports it -- unconditionally, even for
# TP=1 where inter-GPU fd exchange (this module's only job) never runs.
_fd_exchange_stub = types.ModuleType("flashinfer.comm.fd_exchange")


def _unused(*_a, **_kw):
    raise RuntimeError("flashinfer.comm.fd_exchange stubbed out (TP=1, unused)")


_fd_exchange_stub.broadcast_fd = _unused
_fd_exchange_stub.exchange_fds = _unused
sys.modules["flashinfer.comm.fd_exchange"] = _fd_exchange_stub

from transformers.integrations.heterogeneity.configuration_utils import (
    HeterogeneousConfigMixin as _HetCfg,
)

# Fix: vllm 0.27.0's gemma4.py reads legacy flat fields `global_head_dim` /
# `num_global_key_value_heads` for full-attention layers
# (getattr(config, "global_head_dim", config.head_dim), gemma4.py:576+589).
# transformers 5.15.1's Gemma4UnifiedTextConfig moved to the new heterogeneous
# per-layer config system and no longer defines those legacy fields at all, so
# the getattr always silently falls through to the (wrong, sliding-layer)
# default -- every full-attention layer's q_norm/k_norm gets allocated at
# head_dim=256 instead of 512. Root-caused via a weight-loader diagnostic:
# first failure is `layers.11.self_attn.k_norm.weight`, the first
# full-attention layer (of [5,11,17,23,29,35,41,47]) in the checkpoint's
# lexicographic tensor order. Fix: define the legacy properties as a lookup
# into the new per_layer_config view, using the first full_attention layer's
# values (uniform across all full-attention layers in this architecture).
from transformers.models.gemma4_unified.configuration_gemma4_unified import (
    Gemma4UnifiedTextConfig,
)


def _first_full_attn_layer_config(self):
    idx = self.layer_types.index("full_attention")
    return self.per_layer_config[idx]


Gemma4UnifiedTextConfig.global_head_dim = property(
    lambda self: _first_full_attn_layer_config(self).head_dim
)
Gemma4UnifiedTextConfig.num_global_key_value_heads = property(
    lambda self: _first_full_attn_layer_config(self).num_key_value_heads
)

# Diagnostic: wrap default_weight_loader to report the tensor name on a shape
# mismatch (the raw AssertionError doesn't include it). Patched before any
# model file (e.g. gemma4.py) is imported, so `from ... import
# default_weight_loader` picks up this wrapped version.
import vllm.model_executor.model_loader.weight_utils as _wu

_orig_default_weight_loader = _wu.default_weight_loader


def _diag_default_weight_loader(param, loaded_weight):
    if param.size() != loaded_weight.size():
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_locals.get("name", "<unknown>") if frame else "<unknown>"
        print(
            f"[DIAG] shape mismatch on name={name!r}: "
            f"param.size()={tuple(param.size())} "
            f"loaded_weight.size()={tuple(loaded_weight.size())}",
            flush=True,
        )
    return _orig_default_weight_loader(param, loaded_weight)


_wu.default_weight_loader = _diag_default_weight_loader

_HetCfg.allow_global_per_layer_attribute_access = property(
    lambda self: self.__dict__.get("allow_global_per_layer_attribute_access", True),
    lambda self, value: self.__dict__.__setitem__(
        "allow_global_per_layer_attribute_access", value
    ),
)

from vllm.entrypoints.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
