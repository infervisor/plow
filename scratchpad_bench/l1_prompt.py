#!/usr/bin/env python3
"""Longest-match tokenizer over Gemma's vocab — enough for a MEANINGFUL prompt.

The identity gate needs a prompt whose greedy continuation is NOT a constant id.
Random ids give one (the model emits the same token forever), and a constant
stream would pass the comparison even if attention were subtly wrong. This is not
the real BPE merge order, so the ids are not what HF would produce — they are
real vocab entries in a plausible order, which is all the gate needs.
"""
import json, sys

tok = json.load(open("/home/lava/plow/build-amd/g31b-bf16/tokenizer.json"))
vocab = tok["model"]["vocab"]
text = ("▁The capital of France is Paris. The capital of Italy is Rome. "
        "▁The capital of Japan is Tokyo. The capital of Spain is") * 4
text = text.replace(" ", "▁")

ids = [2]  # <bos>
i = 0
while i < len(text):
    for n in range(min(16, len(text) - i), 0, -1):
        piece = text[i:i + n]
        if piece in vocab:
            ids.append(vocab[piece])
            i += n
            break
    else:
        i += 1
n = int(sys.argv[1]) if len(sys.argv) > 1 else len(ids)
print(",".join(str(x) for x in ids[:n]))
