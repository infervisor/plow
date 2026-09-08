#!/usr/bin/env python3
"""Emit three natural-language prompts trimmed to EXACTLY n tokens, as comma-separated ids.

    python3 scripts/mfma_numerics_prompts.py <tokenizer.json> <n> <outdir>

Writes <outdir>/p{0,1,2}_<n>.ids. Real text, not random ids: a greedy stream off random ids
degenerates into a repeat and then "token agreement" is 100% for the wrong reason.
"""
import sys, os
from tokenizers import Tokenizer

TEXTS = [
    """The memory hierarchy of a modern accelerator is not a single number. High-bandwidth
memory sits behind a large last-level cache, which sits behind per-compute-unit vector caches,
and a kernel that streams weights once per token sees almost none of the caching. What it sees
is the raw read rate, divided by however many compute units are actually resident, minus
whatever the address arithmetic and the instruction issue pipe take away. A kernel that is
described as memory bound may in fact be issue bound, and the only way to tell the difference is
to measure the arithmetic separately from the bytes. This is why a roofline drawn from vendor
specifications so often disagrees with a profiler: the specification describes a peak that no
real access pattern reaches, and the profiler describes a pattern whose cost is dominated by
something the roofline never modelled. """,
    """When a language model generates text one token at a time, the arithmetic intensity of the
whole forward pass collapses. Every weight in the network is read from memory and multiplied by
a single vector, so the ratio of floating point operations to bytes moved is about two. Batching
several sequences together is the standard remedy: the weights are read once and multiplied by
several vectors, and the arithmetic intensity rises in proportion to the batch. That only helps
if the multiply itself is cheap. On a part whose vector unit lacks a native mixed-precision dot
product, the emulated multiply costs several instructions per element, and the extra work grows
with the batch while the byte stream does not. The batched decode then becomes compute bound
precisely where it was supposed to become more efficient. """,
    """Numerical reproducibility in a serving engine is a contract, not an accident. A user who
sends the same prompt twice expects the same answer, and a user whose request happens to be
scheduled alongside three others expects the same answer as one who was scheduled alone. That
second guarantee is the harder one: it forbids any kernel whose reduction order depends on how
many sequences share a step. A batched matrix multiply that accumulates each output row into its
own private accumulator satisfies it trivially. One that shares a reduction tree across the
batch does not, and no amount of testing at a fixed batch width will reveal the difference. The
guarantee has to be an argument about the code, checked by a test that varies the batch width
and compares bit patterns. """,
]


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    n = int(sys.argv[2])
    out = sys.argv[3]
    os.makedirs(out, exist_ok=True)
    for i, t in enumerate(TEXTS):
        base = tok.encode(t, add_special_tokens=False).ids
        ids = []
        while len(ids) < n:
            ids.extend(base)
        ids = ids[:n]
        p = os.path.join(out, f"p{i}_{n}.ids")
        open(p, "w").write(",".join(str(v) for v in ids))
        print(p, len(ids))


main()
