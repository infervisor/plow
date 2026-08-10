#!/usr/bin/env python3
"""LONG-PROMPT coherence gate. Exit 0 = PASS, 1 = FAIL.

Two jobs in one request, and the campaign needs both:

 1. COHERENCE at a length the speed bench actually uses. The short "capital of France" gate is
    not enough here: vLLM routes prefills with `prefill_max_seq_len <= topk_tokens` (2048) down a
    DIFFERENT attention path than longer ones, so a short gate certifies a path the long-context
    numbers never touch.

 2. RETRIEVAL, which is what long-context attention can silently lose. A needle is planted EARLY
    and asked for at the END, so answering it requires attending across the whole prompt. A model
    whose attention has degenerated stays fluent and gets this wrong -- which is exactly the
    failure a liveness check cannot see. `LESSONS.md` §9: character-identity gates fire on
    harmless reassociation and say nothing about correctness; this asks a question with an answer.

env: PORT MODEL NTOK
"""
import json, os, random, sys, urllib.request

PORT = os.environ["PORT"]
NTOK = int(os.environ.get("NTOK", "3000"))
BASE = f"http://127.0.0.1:{PORT}"
MODEL = os.environ.get("MODEL", "auto")
if MODEL == "auto":
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
        MODEL = json.load(r)["data"][0]["id"]

NEEDLE = "The Kestrel access code is 7413."
rng = random.Random(20260809)
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
         "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
filler = lambda n: " ".join(rng.choice(WORDS) for _ in range(n))

# Needle at ~10% depth: `LESSONS.md` §9 records that divergence lands ~11% in, so a needle at the
# very front can be answered from a prefix a degraded model still has.
prompt = (filler(int(NTOK * 0.10)) + "\n\n" + NEEDLE + "\n\n" + filler(int(NTOK * 0.90))
          + "\n\nWhat is the Kestrel access code? Reply with just the number.")

body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 200, "temperature": 0}).encode()
req = urllib.request.Request(f"{BASE}/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)["choices"][0]["message"]["content"]
except Exception as e:
    print(f">>> LONG GATE FAIL: request error {e}")
    sys.exit(1)

ok = "7413" in (out or "")
print(f"  long gate ({NTOK} tok): {'PASS' if ok else 'FAIL'} -- {(out or '')[:160]!r}")
sys.exit(0 if ok else 1)
