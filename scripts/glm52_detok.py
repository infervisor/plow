#!/usr/bin/env python3
"""Detokenize a `plowrt amd-bench` greedy-decode id list so a human can read it.

`amd-bench` prints `  [5777, 9125, 1948, ...]` and nothing else, which is the right thing for a
token-identity check and useless for a COHERENCE check — and coherence is the gate a block-fp8
change actually needs, because on GLM-5.2 wrong numerics make the token FASTER (data-dependent
routing collapses the router's top-k and the experts do less work), so a timing A/B cannot detect
them and token identity does not survive a precision change even when it is correct.

  scripts/glm52_detok.py <tokenizer.json> <file-with-the-id-line> [...]

Needs `tokenizers`, which lives in /home/lava/models/oracle_venv (the default dev shell has no
python at all — knob-contract §0a).
"""
import re
import sys

from tokenizers import Tokenizer


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    tok = Tokenizer.from_file(sys.argv[1])
    for path in sys.argv[2:]:
        with open(path) as f:
            text = f.read()
        # The generated stream is the LAST `[...]` of bare integers in the file; the prompt echo and
        # the tracing lines contain brackets too, so match the shape rather than the first hit.
        cands = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", text)
        if not cands:
            print(f"== {path}: no id list found")
            continue
        ids = [int(x) for x in cands[-1].split(",")]
        print(f"== {path}  ({len(ids)} tokens)")
        print(tok.decode(ids))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
