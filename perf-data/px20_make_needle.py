#!/usr/bin/env python3
"""PX-20 needle prompt, as an OpenAI /v1/chat/completions body.

Same haystack construction as PX-8's `make_needle.py` (benign repetitive filler +
the model's chat template), but emitted as a chat REQUEST rather than raw ids, so
the identical prompt can be sent to plowrt and to vLLM through the same client.
Both servers apply the Gemma-4 template themselves (plowrt: `gemma_chat_prompt`
in serve/chat.rs; vLLM: the checkpoint's chat_template.jinja), so the token count
printed here is what each engine will prefill, up to template-rendering noise.

  px20_make_needle.py --tokens 66901 --out /tmp/px20/needle67k.json
"""
import argparse, json

FILLER = ("The quarterly logistics report describes routine warehouse operations across "
          "regional distribution centers. Inventory levels remained within expected "
          "tolerances and no exceptions were escalated. ")
NEEDLE = "The secret access code for the Zurich vault is PELICAN-7734. "
QUESTION = ("\n\nQuestion: What is the secret access code for the Zurich vault? "
            "Answer with just the code.")
SECRET = "PELICAN-7734"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/root/gemma-4-12B-it")
    ap.add_argument("--tokens", type=int, required=True, help="target TOTAL prompt tokens")
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--model", default="MODEL")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if getattr(tok, "chat_template", None) is None:
        tok.chat_template = open("/root/gemma-4-12B-it/chat_template.jinja").read()

    def render(n_sec):
        at = max(1, int(n_sec * a.depth))
        body = FILLER * at + NEEDLE + FILLER * (n_sec - at) + QUESTION
        text = tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)
        return body, at, len(tok(text, add_special_tokens=False).input_ids)

    per = len(tok(FILLER, add_special_tokens=False).input_ids)
    n_sec = max(1, (a.tokens - 128) // per)
    body, at, n = render(n_sec)
    while n > a.tokens and n_sec > 1:
        n_sec -= max(1, (n - a.tokens) // per)
        body, at, n = render(n_sec)
    while True:  # walk back up so the prompt lands just UNDER the target
        b2, at2, n2 = render(n_sec + 1)
        if n2 > a.tokens:
            break
        n_sec, body, at, n = n_sec + 1, b2, at2, n2

    req = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "stream": False,
           "messages": [{"role": "user", "content": body}]}
    with open(a.out, "w") as f:
        json.dump(req, f)
    print(f"{a.out}: {n} templated prompt tokens, {n_sec} filler sections, "
          f"needle at section {at} (depth {at/n_sec:.2f}), must answer {SECRET!r}")


if __name__ == "__main__":
    main()
