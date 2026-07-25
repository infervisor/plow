#!/usr/bin/env python3
"""Materialize the exact seed-0 vLLM RandomDataset prompts as int32 token IDs.

This is for standalone Plow comparisons with `bench_vllm_cuda.sh`, whose B1
baseline uses three random prompts, seed 0, and a zero input-length range.

Usage:
  make_vllm_random_ids.py MODEL_DIR OUT_DIR [CONTEXT ...]
"""

import argparse
import os
import struct

from transformers import AutoTokenizer
from vllm.benchmarks.datasets import RandomDataset


DEFAULT_CONTEXTS = (1024, 4096, 16384, 32768, 65536, 98304, 131072)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("out_dir")
    parser.add_argument("contexts", type=int, nargs="*", default=DEFAULT_CONTEXTS)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    for context in args.contexts:
        # The vLLM CLI creates a fresh dataset for each benchmark invocation,
        # so reset seed 0 for each context too.
        requests = RandomDataset(random_seed=0).sample(
            tokenizer,
            num_requests=3,
            range_ratio=0.0,
            input_len=context,
            output_len=128,
        )
        for index, request in enumerate(requests):
            ids = tokenizer(request.prompt, add_special_tokens=False).input_ids
            if len(ids) != context or request.prompt_len != context:
                raise RuntimeError(
                    f"ctx={context} prompt={index}: expected {context} IDs, "
                    f"got request={request.prompt_len}, encoded={len(ids)}"
                )
            path = os.path.join(args.out_dir, f"ids_{context}_p{index}.bin")
            with open(path, "wb") as out:
                out.write(struct.pack(f"<{len(ids)}i", *ids))
            print(f"{path}: {len(ids)} ids, first={ids[:4]}")


if __name__ == "__main__":
    main()
