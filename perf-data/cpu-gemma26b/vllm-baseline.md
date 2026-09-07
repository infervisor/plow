# Why there is no vLLM baseline for Gemma-4-26B-A4B, with the actual mechanism

The consolidated summary previously said vLLM "has no 4-bit path for it". **That was wrong.**
vLLM 0.28.0's CPU backend has four quantized MoE expert paths for x86, all AMX-gated, and this box
has AMX. The real blocker is narrower and more interesting: every one of those paths requires a
SILU-family activation, and Gemma-4's MoE uses GELU-tanh.

Traced statically in the installed wheel
(`/home/lava/vllm-cpu/venv/lib/python3.12/site-packages/vllm`), no model load required.

## The chain

1. **The architecture is supported.** `Gemma4ForConditionalGeneration` and `Gemma4ForCausalLM` are
   both in `ModelRegistry.get_supported_archs()` (378 archs). Arch support was never the problem.

2. **vLLM builds the Gemma-4 MoE with GELU-tanh.** `model_executor/models/gemma4.py:368` passes
   `activation="gelu_tanh"`, matching the checkpoint's `hidden_activation: "gelu_pytorch_tanh"`.

3. **Every x86 quantized CPU expert path rejects that activation.** From
   `model_executor/layers/fused_moe/experts/cpu_moe.py`:

   | class | line | device gate | `_supports_activation` |
   |---|---|---|---|
   | `CPUExpertsFp8` | 551 | x86 + AMX | SILU only |
   | `CPUExpertsMxfp4` | 709 | x86 + AMX | SILU, SWIGLUOAI |
   | `CPUExpertsInt4` | 910 | x86 + AMX | SILU only |
   | `CPUExpertsInt8` | 1059 | x86 + AMX | SILU only |
   | `ArmCPUExpertsInt8` | 1228 | **ARM** + `cpu_fused_moe_int8` | SILU, SWIGLUOAI, GELU, GELU_TANH |
   | `X86CPUUnquantizedExperts` | 431 | x86 + AMX | SILU, SWIGLUOAI, GELU, GELU_TANH |

   The one CPU int8 path that does accept GELU-tanh is ARM-only, and its kernel is absent from this
   build anyway (`hasattr(torch.ops._C, "cpu_fused_moe_int8")` is `False`; `cpu_fused_moe` is `True`).

4. **The only path left is unquantized.** `CPUUnquantizedExperts._supports_quant_scheme` returns
   true only for `(weight_key, activation_key) == (None, None)`, so the GELU-capable x86 path is
   bf16 by construction.

5. **bf16 does not fit.** Parsed from the safetensors headers:

   | group | tensors | GiB |
   |---|---|---|
   | MoE experts | 60 | 42.54 |
   | language_model other | 596 | 3.09 |
   | embed_tokens (tied lm_head) | 1 | 1.38 |
   | vision_tower | 356 | 1.07 |
   | **total** | **1013** | **48.07** |

   Text-only is **47.00 GiB** on a box with `MemTotal` 58.85 GiB, before the torch runtime, the KV
   cache, and activations. Serving text-only does not help: the vision tower is just 1.07 GiB.
   This matches the observed failure — engine core dies during init at 180 s, even at 2048 context
   with one sequence.

## What this changes

Nothing about plow, and nothing about the box size argument's conclusion — but the *reason* matters
for anyone who revisits this:

* A bigger box would let vLLM run this model **only in bf16**. It would still refuse every
  quantized CPU expert kernel, because the gate is the activation, not memory.
* So the comparison plow wins here is not "plow quantizes and vLLM cannot fit"; it is "vLLM CPU has
  no quantized MoE kernel for a GELU-family router on any x86 box". plow serves the same model from
  a 13 GB MXFP4 twin at ~21 GB resident.
* `supported_quantization` is `[]` for `CpuPlatform`, and `platforms/interface.py:966` reads
  `if cls.supported_quantization and quant not in cls.supported_quantization`. The empty list means
  **no platform-level gate at all**, not "nothing supported". Do not read that attribute as a
  capability list.

## Reproduce

```sh
bash    perf-data/tools/vllm-cpu-capability-probe.sh   # op presence + platform capability
python3 perf-data/tools/safetensors-tower-bytes.py     # per-tower bytes from safetensors headers
```

Neither loads the model; both run in seconds. The byte accounting needs no torch or numpy — it
parses the safetensors JSON header directly, which is why it runs under the system python.

The activation gates are worth re-reading rather than trusting this table, because they are the
kind of thing upstream changes: `_supports_activation` on each `CPUExperts*` class in
`vllm/model_executor/layers/fused_moe/experts/cpu_moe.py`. Checked against vLLM 0.28.0.
