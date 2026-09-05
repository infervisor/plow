#!/usr/bin/env python3
"""Capture dense next-token logits from an unmodified vLLM installation.

The input is a JSON object with a ``cases`` array.  Each case contains a stable
``id`` and explicit ``prompt_token_ids``.  vLLM's public full-logprob path is
run in ``raw_logits`` mode, so this does not patch the model or timed kernels.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--max-output-tokens", type=int, default=1)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--disable-cuda-graphs", action="store_true")
    p.add_argument("--language-model-only", action="store_true")
    p.add_argument("--quantization", choices=["fp8"])
    return p.parse_args()


def case_id(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe case id: {value!r}")
    return value


def prompt_digest(ids):
    a = np.asarray(ids, dtype="<u4")
    return hashlib.sha256(a.tobytes()).hexdigest()


def suppression_metadata(model_config, vocab_size):
    config = model_config.try_get_generation_config() or {}
    ids = config.get("suppress_tokens") or []
    if not isinstance(ids, list) or any(
        type(token) is not int or not 0 <= token < vocab_size for token in ids
    ):
        raise ValueError("generation_config.suppress_tokens must contain valid vocabulary IDs")
    return {
        "source": "vllm.model_config.try_get_generation_config().suppress_tokens",
        "token_ids": sorted(set(ids)),
        "allowed_nonfinite": "negative_infinity_only_at_declared_ids",
        "raw_layout": "full_vocabulary_in_original_token_id_order",
        "repeat_metrics_exclude_token_ids": sorted(set(ids)),
    }


def dense_scores(position, vocab_size, diagnostic_prefix=None, suppressed_ids=()):
    # Offline outputs use either the compatibility dict or FlatLogprobs.  Both
    # expose token IDs, unlike the OpenAI JSON surface where decoded-token keys
    # can collide.
    if hasattr(position, "token_ids"):
        pairs = zip(position.token_ids, position.logprobs)
    else:
        pairs = ((token_id, entry.logprob) for token_id, entry in position.items())
    out = np.full(vocab_size, np.nan, dtype=np.float32)
    seen = np.zeros(vocab_size, dtype=bool)
    for token_id, score in pairs:
        token_id = int(token_id)
        if 0 <= token_id < vocab_size:
            out[token_id] = float(score)
            seen[token_id] = True
    missing = int((~seen).sum())
    allowed = np.zeros(vocab_size, dtype=bool)
    allowed[list(suppressed_ids)] = True
    invalid = ~np.isfinite(out) & ~(allowed & np.isneginf(out))
    if missing or invalid.any():
        diagnostic = {
            "vocab_size": vocab_size,
            "missing_indices": np.flatnonzero(~seen).tolist(),
            "nan_indices": np.flatnonzero(np.isnan(out) & seen).tolist(),
            "negative_inf_indices": np.flatnonzero(np.isneginf(out)).tolist(),
            "positive_inf_indices": np.flatnonzero(np.isposinf(out)).tolist(),
            "finite_count": int(np.isfinite(out).sum()),
        }
        if diagnostic_prefix is not None:
            Path(f"{diagnostic_prefix}.invalid.f32").write_bytes(out.astype("<f4").tobytes())
            Path(f"{diagnostic_prefix}.invalid.json").write_text(json.dumps(diagnostic, indent=2))
        counts = {key: len(value) for key, value in diagnostic.items() if isinstance(value, list)}
        raise RuntimeError(f"vLLM returned invalid logits: {counts}")
    return out


def repeat_metrics(current, prior, suppressed_ids=()):
    keep = np.ones(len(current), dtype=bool)
    keep[list(suppressed_ids)] = False
    a = current[keep].astype(np.float64)
    b = prior[keep].astype(np.float64)
    if not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("repeat metrics require finite unsuppressed logits")
    a -= a.mean()
    b -= b.mean()
    delta = a - b
    head = np.argsort(b)[-64:]
    return {
        "full_row_centered_rel_l2": float(
            np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30)
        ),
        "reference_head64_centered_rel_l2": float(
            np.linalg.norm(delta[head]) / max(np.linalg.norm(b[head]), 1e-30)
        ),
        "centered_max_abs": float(np.max(np.abs(delta))),
        "top64_overlap": len(set(np.argsort(a)[-64:]) & set(head)) / 64,
        "same_argmax": bool(a.argmax() == b.argmax()),
    }


def required_model_length(cases, output_tokens, requested=None):
    if output_tokens < 1:
        raise ValueError("max-output-tokens must be positive")
    lengths = [len(case["prompt_token_ids"]) for case in cases]
    if not lengths or min(lengths) < 1:
        raise ValueError("cases must contain non-empty prompts")
    return max(max(lengths) + output_tokens, requested or 0)


def generation_rows(cid, prompt_ids, generated_ids, output_tokens):
    if output_tokens < 1 or len(generated_ids) != output_tokens:
        raise ValueError("generated token count must match max-output-tokens")
    prefix = list(prompt_ids)
    rows = []
    for step, token in enumerate(generated_ids):
        rows.append({
            "id": cid if output_tokens == 1 else f"{cid}.step{step:04d}",
            "prompt_len": len(prefix),
            "prompt_sha256_u32le": prompt_digest(prefix),
            "sampled_token_id": int(token),
            "request_case_id": cid,
            "generation_step": step,
            "execution_phase": "prefill_output" if step == 0 else "decode_output",
        })
        prefix.append(int(token))
    return rows


def main():
    args = parse_args()
    request = json.loads(args.cases.read_text())
    cases = request.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases JSON needs a non-empty `cases` array")
    max_len = required_model_length(cases, args.max_output_tokens, args.max_model_len)
    case_ids = [case_id(case["id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")

    from vllm import LLM, SamplingParams, __version__ as vllm_version

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=args.trust_remote_code,
        skip_tokenizer_init=True,
        max_model_len=max_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        language_model_only=args.language_model_only,
        quantization=args.quantization,
        **({"compilation_config": {"cudagraph_mode": "NONE"}}
           if args.disable_cuda_graphs else {}),
    )
    effective_compile = llm.llm_engine.vllm_config.compilation_config
    vocab_size = int(llm.model_config.get_vocab_size())
    suppression = suppression_metadata(llm.model_config, vocab_size)
    suppressed_ids = suppression["token_ids"]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_output_tokens,
        ignore_eos=True,
        logprobs=-1,
        flat_logprobs=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "producer": "vllm-public-raw-logits",
        "vllm_version": vllm_version,
        "model": os.path.realpath(args.model),
        "tensor_parallel_size": args.tp,
        "dtype": "float32",
        "vocab_size": vocab_size,
        "logprobs_mode": "raw_logits",
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "max_model_len": max_len,
        "max_output_tokens": args.max_output_tokens,
        "enforce_eager": args.enforce_eager,
        "disable_cuda_graphs": args.disable_cuda_graphs,
        "effective_compilation_config": {
            "mode": getattr(effective_compile.mode, "name", str(effective_compile.mode)),
            "cudagraph_mode": getattr(effective_compile.cudagraph_mode, "name", str(effective_compile.cudagraph_mode)),
            "backend": effective_compile.backend,
            "custom_ops": list(effective_compile.custom_ops),
        },
        "language_model_only": args.language_model_only,
        "quantization": args.quantization,
        "hf_vocab_size": llm.model_config.hf_text_config.vocab_size,
        "final_logit_softcapping": getattr(llm.model_config.hf_text_config, "final_logit_softcapping", None),
        "suppression": suppression,
        "requests": [],
        "cases": [],
        "repeat_checks": [],
        "invalid_cases": [],
    }
    (args.output / "invocation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    seen = {}
    for raw_case in cases:
        cid = case_id(raw_case["id"])
        ids = [int(x) for x in raw_case["prompt_token_ids"]]
        if not ids:
            raise ValueError(f"case {cid} has an empty prompt")
        result = llm.generate(
            {"prompt_token_ids": ids}, sampling, use_tqdm=False
        )[0]
        completion = result.outputs[0]
        generated_ids = [int(token) for token in completion.token_ids]
        manifest["requests"].append({
            "id": cid,
            "prompt_token_ids": ids,
            "prompt_sha256_u32le": prompt_digest(ids),
            "generated_token_ids": generated_ids,
            "finish_reason": completion.finish_reason,
            "request_id": result.request_id,
        })
        rows = generation_rows(cid, ids, generated_ids, args.max_output_tokens)
        if not completion.logprobs or len(completion.logprobs) != len(rows):
            raise RuntimeError(f"case {cid}: missing per-token logprobs")
        for row, position in zip(rows, completion.logprobs):
            row_id = row["id"]
            try:
                scores = dense_scores(position, vocab_size, args.output / row_id, suppressed_ids)
            except RuntimeError as error:
                manifest["invalid_cases"].append({"id": row_id, "error": str(error)})
                continue
            filename = f"logits_{row_id}.f32"
            scores.astype("<f4", copy=False).tofile(args.output / filename)
            sha = row["prompt_sha256_u32le"]
            if sha in seen:
                prior_id, prior_scores = seen[sha]
                manifest["repeat_checks"].append({
                    "first_case": prior_id,
                    "repeat_case": row_id,
                    "prompt_sha256_u32le": sha,
                    **repeat_metrics(scores, prior_scores, suppressed_ids),
                })
            else:
                seen[sha] = (row_id, scores.copy())
            manifest["cases"].append({
                **row,
                "file": filename,
                "negative_inf_token_ids": np.flatnonzero(np.isneginf(scores)).tolist(),
            })
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["invalid_cases"]:
        raise RuntimeError(f"{len(manifest['invalid_cases'])} cases rejected; see manifest.json")


if __name__ == "__main__":
    main()
