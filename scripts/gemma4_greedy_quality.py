#!/usr/bin/env python3
"""Greedy-completion capture and comparison for a precision change on one engine.

`capture` sends a fixed corpus of natural-text prompts at temperature 0 and records
the completion text. `compare` re-tokenizes two captures with the model's own
tokenizer and reports, per prompt, the first position where the two token sequences
diverge and the fraction of positions that agree. Text is the only common surface:
plowrt refuses `logprobs`, so per-token ids cannot be read back from both engines.

Two prompt modes, because they measure different things:

* `chat` (default) puts the excerpt in a user turn through the server's own chat
  template and asks a question about it, so the model generates ordinary prose.
* `raw` feeds the excerpt to `/v1/completions` with `add_special_tokens=false`,
  which is what the throughput harness sends. It is exact on prompt length and
  useless for quality: with no BOS and no turn structure this instruction-tuned
  checkpoint collapses into single-token repetition within a few tokens, and the
  agreement number then measures which degenerate attractor each arm fell into.
"""
import argparse
import hashlib
import json
import statistics
import sys
import urllib.request

from tokenizers import Tokenizer


QUESTION = ("\n\nUsing only the text above, explain what it specifies and why. "
            "Write plain prose.")


def build_prompts(tokenizer_path, sources, lengths, per_length, exact):
    tok = Tokenizer.from_file(tokenizer_path)
    text = "\n\n".join(open(p, encoding="utf-8", errors="replace").read() for p in sources)
    ids = tok.encode(text, add_special_tokens=False).ids
    prompts = []
    cursor = 0
    for length in lengths:
        for index in range(per_length):
            # Re-encoding a decoded window is not length-preserving in general, so
            # trim until the re-encoded window is exactly `length`. Only the raw mode
            # needs this; a chat prompt's served length includes the template.
            window = length
            while True:
                if cursor + window > len(ids):
                    sys.exit(f"corpus exhausted at length {length}")
                candidate = tok.decode(ids[cursor:cursor + window])
                served = len(tok.encode(candidate, add_special_tokens=False).ids)
                if served == length or not exact:
                    break
                window += length - served
                if window <= 0:
                    sys.exit(f"cannot build a {length}-token prompt")
            prompts.append({"length": length, "index": index, "text": candidate,
                            "sha256": hashlib.sha256(candidate.encode()).hexdigest()})
            cursor += window
    return prompts


def post(url, path, body, timeout=900):
    request = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def cmd_capture(args):
    exact = args.mode == "raw"
    prompts = build_prompts(args.tokenizer, args.corpus, args.lengths,
                            args.per_length, exact)
    model = json.load(urllib.request.urlopen(args.url + "/v1/models"))["data"][0]["id"]
    with open(args.out, "w") as log:
        for prompt in prompts:
            if exact:
                result = post(args.url, "/v1/completions",
                              {"model": model, "prompt": prompt["text"],
                               "add_special_tokens": False, "temperature": 0,
                               "max_tokens": args.max_tokens, "ignore_eos": True,
                               "stream": False})
                text = result["choices"][0]["text"]
            else:
                result = post(args.url, "/v1/chat/completions",
                              {"model": model, "temperature": 0,
                               "max_tokens": args.max_tokens, "stream": False,
                               "messages": [{"role": "user",
                                             "content": prompt["text"] + QUESTION}]})
                text = result["choices"][0]["message"]["content"]
            usage = result["usage"]
            if exact:
                assert usage["prompt_tokens"] == prompt["length"], (prompt["length"], usage)
                assert usage["completion_tokens"] == args.max_tokens, usage
            record = {"label": args.label, "model": model, "mode": args.mode,
                      "length": prompt["length"], "index": prompt["index"],
                      "prompt_sha256": prompt["sha256"], "completion": text,
                      "usage": usage}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"{args.label} len={prompt['length']} i={prompt['index']} "
                  f"prompt_tokens={usage['prompt_tokens']} "
                  f"completion_tokens={usage['completion_tokens']}", flush=True)


def cmd_compare(args):
    tok = Tokenizer.from_file(args.tokenizer)
    def load(path):
        rows = {}
        for line in open(path):
            r = json.loads(line)
            rows[(r["length"], r["index"])] = r
        return rows
    left, right = load(args.left), load(args.right)
    shared = sorted(set(left) & set(right))
    if not shared:
        sys.exit("captures share no prompts")
    per_length = {}
    rows = []
    for key in shared:
        a, b = left[key], right[key]
        assert a["prompt_sha256"] == b["prompt_sha256"], key
        assert a["usage"]["prompt_tokens"] == b["usage"]["prompt_tokens"], key
        assert a.get("mode") == b.get("mode"), key
        ta = tok.encode(a["completion"], add_special_tokens=False).ids
        tb = tok.encode(b["completion"], add_special_tokens=False).ids
        n = min(len(ta), len(tb))
        first = next((i for i in range(n) if ta[i] != tb[i]), None)
        if first is None and len(ta) != len(tb):
            first = n
        agree = sum(1 for i in range(n) if ta[i] == tb[i])
        compared = max(len(ta), len(tb))
        row = {"length": key[0], "index": key[1], "tokens_left": len(ta),
               "tokens_right": len(tb), "agree": agree, "compared": compared,
               "agreement": agree / compared if compared else 1.0,
               "first_divergence": first, "identical_text": a["completion"] == b["completion"]}
        rows.append(row)
        per_length.setdefault(key[0], []).append(row)
    summary = {"left": args.left, "right": args.right, "prompts": len(rows),
               "per_length": {}, "rows": rows}
    for length, group in sorted(per_length.items()):
        diverged = [r["first_divergence"] for r in group if r["first_divergence"] is not None]
        summary["per_length"][str(length)] = {
            "prompts": len(group),
            "token_agreement": sum(r["agree"] for r in group) / sum(r["compared"] for r in group),
            "identical_text": sum(r["identical_text"] for r in group),
            "median_first_divergence": statistics.median(diverged) if diverged else None,
            "min_first_divergence": min(diverged) if diverged else None,
        }
    summary["overall_token_agreement"] = (
        sum(r["agree"] for r in rows) / sum(r["compared"] for r in rows))
    json.dump(summary, open(args.out, "w"), indent=2)
    for length, stats in summary["per_length"].items():
        print(f"len={length:>7} agreement={stats['token_agreement']:.4f} "
              f"identical_text={stats['identical_text']}/{stats['prompts']} "
              f"median_first_divergence={stats['median_first_divergence']}", flush=True)
    print(f"overall token agreement {summary['overall_token_agreement']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture")
    cap.add_argument("--url", required=True)
    cap.add_argument("--label", required=True)
    cap.add_argument("--out", required=True)
    cap.add_argument("--tokenizer", required=True)
    cap.add_argument("--corpus", nargs="+", required=True)
    cap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
    cap.add_argument("--per-length", type=int, default=3)
    cap.add_argument("--max-tokens", type=int, default=64)
    cap.add_argument("--mode", choices=("chat", "raw"), default="chat")
    cap.set_defaults(func=cmd_capture)

    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--left", required=True)
    cmp_.add_argument("--right", required=True)
    cmp_.add_argument("--tokenizer", required=True)
    cmp_.add_argument("--out", required=True)
    cmp_.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
