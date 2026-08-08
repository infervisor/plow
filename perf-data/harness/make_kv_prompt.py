#!/usr/bin/env python3
"""Build a real-text prompt (.ids, raw int32 LE) for the P10 KV-0 dump.

The existing perf prompts are wrong for a compressibility audit: measure_tpot
prompts are one phrase REPEATED (repetition inflates KV compressibility) and
RandomDataset ids are token noise (unrepresentative activations). This script
tokenizes genuine, non-repetitive text gathered from the repo:

  --corpus prose: perf-data/*.md + README + docs (technical English)
  --corpus code  : runtime/nvidia/* + crates/**/*.rs (source code)

Run under the vllm venv python (its transformers has the gemma-4 tokenizer):
  /workspace/venvs/vllm/bin/python make_kv_prompt.py \
      --model /workspace/models/gemma-4-12B-it --repo /root/plow \
      --corpus prose --tokens 24576 --out /dev/shm/kv0/prose.ids
"""
import argparse, glob, os, struct, sys


def gather(repo, corpus):
    if corpus == "prose":
        pats = ["perf-data/*.md", "README.md", "docs/*.md", "docs/arch/*.md"]
    elif corpus == "code":
        pats = ["runtime/nvidia/*.cuh", "runtime/nvidia/*.cu",
                "crates/plowc/src/bin/*.rs", "crates/plowrt/src/**/*.rs"]
    else:
        sys.exit(f"unknown corpus {corpus}")
    files = []
    for p in pats:
        files += sorted(glob.glob(os.path.join(repo, p), recursive=True))
    parts = []
    for f in files:
        try:
            parts.append(open(f, encoding="utf-8", errors="ignore").read())
        except OSError:
            pass
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--corpus", required=True, choices=["prose", "code"])
    ap.add_argument("--tokens", type=int, default=24576)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    text = gather(a.repo, a.corpus)
    ids = tok(text, add_special_tokens=True).input_ids
    if len(ids) < a.tokens:
        sys.exit(f"corpus too small: {len(ids)} < {a.tokens} tokens")
    ids = ids[: a.tokens]
    with open(a.out, "wb") as f:
        f.write(struct.pack(f"<{len(ids)}i", *ids))
    # distinctness check: a repeated-phrase prompt has a tiny unique-ngram share
    uniq8 = len({tuple(ids[i : i + 8]) for i in range(0, len(ids) - 8, 8)})
    print(f"{a.out}: {len(ids)} tokens, unique-8gram share "
          f"{uniq8 / max(1, (len(ids) - 8) // 8):.3f}")


if __name__ == "__main__":
    main()
