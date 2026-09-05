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
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--language-model-only", action="store_true")
    return p.parse_args()


def case_id(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe case id: {value!r}")
    return value


def prompt_digest(ids):
    a = np.asarray(ids, dtype="<u4")
    return hashlib.sha256(a.tobytes()).hexdigest()


def dense_scores(position, vocab_size, diagnostic_prefix=None):
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
    if missing or not np.isfinite(out).all():
        diagnostic = {
            "vocab_size": vocab_size,
            "missing_indices": np.flatnonzero(~seen).tolist(),
            "nan_indices": np.flatnonzero(np.isnan(out) & seen).tolist(),
            "negative_inf_indices": np.flatnonzero(np.isneginf(out)).tolist(),
            "positive_inf_indices": np.flatnonzero(np.isposinf(out)).tolist(),
            "finite_count": int(np.isfinite(out).sum()),
        }
        if diagnostic_prefix is not None:
            diagnostic_prefix.with_suffix(".invalid.f32").write_bytes(out.astype("<f4").tobytes())
            diagnostic_prefix.with_suffix(".invalid.json").write_text(json.dumps(diagnostic, indent=2))
        counts = {key: len(value) for key, value in diagnostic.items() if isinstance(value, list)}
        raise RuntimeError(f"vLLM returned invalid logits: {counts}")
    return out


def repeat_metrics(current, prior):
    a = current.astype(np.float64)
    b = prior.astype(np.float64)
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


def main():
    args = parse_args()
    request = json.loads(args.cases.read_text())
    cases = request.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases JSON needs a non-empty `cases` array")
    max_len = max(max(len(c["prompt_token_ids"]) for c in cases) + 1, args.max_model_len or 0)

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
    )
    vocab_size = int(llm.model_config.get_vocab_size())
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
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
        "max_model_len": max_len,
        "enforce_eager": args.enforce_eager,
        "language_model_only": args.language_model_only,
        "hf_vocab_size": llm.model_config.hf_text_config.vocab_size,
        "final_logit_softcapping": getattr(llm.model_config.hf_text_config, "final_logit_softcapping", None),
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
        if not completion.logprobs or len(completion.logprobs) != 1:
            raise RuntimeError(f"case {cid}: missing one-step logprobs")
        try:
            scores = dense_scores(completion.logprobs[0], vocab_size, args.output / cid)
        except RuntimeError as error:
            manifest["invalid_cases"].append({"id": cid, "error": str(error)})
            continue
        filename = f"logits_{cid}.f32"
        scores.astype("<f4", copy=False).tofile(args.output / filename)
        sha = prompt_digest(ids)
        if sha in seen:
            prior_id, prior_scores = seen[sha]
            manifest["repeat_checks"].append(
                {
                    "first_case": prior_id,
                    "repeat_case": cid,
                    "prompt_sha256_u32le": sha,
                    **repeat_metrics(scores, prior_scores),
                }
            )
        else:
            seen[sha] = (cid, scores.copy())
        manifest["cases"].append(
            {
                "id": cid,
                "file": filename,
                "prompt_len": len(ids),
                "prompt_sha256_u32le": sha,
                "sampled_token_id": int(completion.token_ids[0]),
            }
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["invalid_cases"]:
        raise RuntimeError(f"{len(manifest['invalid_cases'])} cases rejected; see manifest.json")


if __name__ == "__main__":
    main()
