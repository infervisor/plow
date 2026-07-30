#!/usr/bin/env python3
# kimi_k3_tokenizer.py — build a real `tokenizer.json` for moonshotai/Kimi-K3.
#
# WHY THIS EXISTS.  K3 ships `tiktoken.model` + a Python `tokenization_kimi.py`, and NO
# `tokenizer.json`.  plowrt's AMD serve path binds the Rust `tokenizers` crate (plowrt/Cargo.toml:52)
# and REFUSES to serve a model whose tokenizer is the byte fallback (`main.rs:1397`) — a real
# checkpoint driven through byte-fallback ids produces fluent-looking GARBAGE rather than an error,
# so the refusal is correct and the fix is to supply the file, not to loosen the check.
#
# WHAT A TIKTOKEN ENCODING ACTUALLY IS, and hence what has to be reconstructed:
#
#   * `tiktoken.model` is `base64(token_bytes) rank` per line — 163,584 lines here, ranks dense
#     over 0..163583.  It is a RANK TABLE, not a merge list: tiktoken merges by "lowest rank of
#     the concatenation", never consulting a merge rule.  HF BPE needs the MERGES, so each
#     multi-byte token's rank-ordered split has to be recovered (`recover_merge`).
#
#   * The 256 specials live ABOVE the base vocab at 163584..163839 and are NOT in the file.
#     `tokenization_kimi.py:101` builds them as `added_tokens_decoder[i]` where present and
#     `<|reserved_token_{i}|>` otherwise — keyed on the ABSOLUTE id, not an offset.  163584+256
#     = 163840 = the emitter's `vocab` (`crates/devgen/src/k3.rs`), which is the check that the
#     count is right rather than merely plausible.
#
#   * tiktoken splits TEXT with `pat_str` and then runs BPE over each piece's UTF-8 BYTES.  The HF
#     equivalent is Split(pattern, "isolated") followed by a ByteLevel that does NOT re-split
#     (`use_regex=False`) and does NOT add a prefix space — with the GPT-2 byte->unicode table
#     applied to every vocab key so the two agree on what a "character" is.
#
# VERIFICATION IS THE POINT.  A converter that is 99.9% right is worse than none: it produces
# plausible text and a silent quality regression.  `--verify` encodes a corpus through BOTH the
# real tiktoken Encoding and the emitted tokenizer.json and requires EXACT id-sequence equality,
# then round-trips the decode.  Non-zero exit on any mismatch.
#
# Usage:
#   python3 scripts/kimi_k3_tokenizer.py --model <snapshot> --out <snapshot-or-dir> --verify
import argparse, base64, json, os, sys


# ---------------------------------------------------------------- byte <-> unicode (GPT-2 table)
def byte_to_unicode():
    """The GPT-2 table: every byte to a printable, non-space codepoint.

    HF stores BPE vocab keys as STRINGS, so raw bytes have to be carried as characters that survive
    JSON and never collide with whitespace the pre-tokenizer might act on.  188 bytes map to
    themselves; the remaining 68 are lifted to U+0100.. in order.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(
        range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = byte_to_unicode()


def enc_key(tok: bytes) -> str:
    return "".join(B2U[b] for b in tok)


# ---------------------------------------------------------------- merges
def recover_merge(ranks, token, rank):
    """The pair `token` was formed from, by replaying BPE with only STRICTLY EARLIER merges.

    tiktoken never stores this. The reconstruction is exact rather than heuristic: a token of rank
    r is by construction the concatenation of the two pieces that survive when every merge of rank
    < r has been applied and no merge of rank >= r is allowed. Replaying under that ceiling
    therefore has to terminate at exactly two parts; anything else means the table is not a
    well-formed BPE and we refuse rather than guess.
    """
    parts = [bytes([b]) for b in token]
    while True:
        best_i, best_rank = None, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < rank and (best_rank is None or r < best_rank):
                best_i, best_rank = i, r
        if best_i is None:
            break
        parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
    return parts


def load_tiktoken(path):
    ranks = {}
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b64, rank = line.split()
            ranks[base64.b64decode(b64)] = int(rank)
    return ranks


# ---------------------------------------------------------------- specials
def special_tokens(model_dir, n_base, n_reserved=256):
    """ids n_base..n_base+n_reserved-1, exactly as `tokenization_kimi.py:101` builds them."""
    cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    named = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        for i, e in (cfg.get("added_tokens_decoder") or {}).items():
            named[int(i)] = e
    out = []
    for i in range(n_base, n_base + n_reserved):
        e = named.get(i)
        content = e["content"] if e else f"<|reserved_token_{i}|>"
        out.append({
            "id": i,
            "content": content,
            # `special=True` is what makes the Rust tokenizer refuse to split these out of user
            # text unless asked; mirror the checkpoint's own flag where it states one. The
            # reserved filler is special by construction.
            "single_word": bool(e.get("single_word", False)) if e else False,
            "lstrip": bool(e.get("lstrip", False)) if e else False,
            "rstrip": bool(e.get("rstrip", False)) if e else False,
            "normalized": bool(e.get("normalized", False)) if e else False,
            "special": bool(e.get("special", True)) if e else True,
        })
    return out


PAT = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


def build(model_dir):
    ranks = load_tiktoken(os.path.join(model_dir, "tiktoken.model"))
    n_base = len(ranks)
    if sorted(ranks.values()) != list(range(n_base)):
        sys.exit("tiktoken.model ranks are not dense over 0..n-1; refusing to guess")

    by_rank = sorted(ranks.items(), key=lambda kv: kv[1])
    vocab = {enc_key(tok): r for tok, r in by_rank}
    merges = []
    for tok, r in by_rank:
        if len(tok) == 1:
            continue
        parts = recover_merge(ranks, tok, r)
        if len(parts) != 2:
            sys.exit(f"rank {r} ({tok!r}) replayed to {len(parts)} parts, not 2 — table is not BPE")
        merges.append([enc_key(parts[0]), enc_key(parts[1])])

    added = special_tokens(model_dir, n_base)
    for a in added:
        vocab[a["content"]] = a["id"]

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        # Split THEN ByteLevel(use_regex=False): the split is tiktoken's own pat_str, and the
        # ByteLevel must not impose GPT-2's regex on top of it or the two tokenizers disagree on
        # every piece boundary. add_prefix_space=False for the same reason — tiktoken adds none.
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": PAT}, "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                 "use_regex": False},
            ],
        },
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False,
                           "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True,
                    "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }, n_base


# ---------------------------------------------------------------- verify
CORPUS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n    return x ** 2  # square\n",
    "  leading and   internal   spaces\t\tand tabs\n\n\n",
    "3.14159 and 42 and 1000000 and 007",
    "你好，世界。这是一个测试。",
    "日本語のテキストです。カタカナもひらがなも。",
    "한국어 텍스트입니다.",
    "русский текст здесь",
    "emoji: \U0001f600\U0001f680\U0001f9e0 and combining: éà",
    "It's a test, isn't it? I'd say we've done it. You'll see. They're here.",
    "IT'S ALL CAPS AND It's Mixed Case",
    "mixed 中文 and English 混合 text 123 测试",
    "a" * 300,
    "\n\n\n\t\t  \r\n mixed whitespace \r\n\r\n",
    "https://example.com/path?query=1&other=2#frag",
    '{"json": [1, 2, {"nested": true}], "unicode": "\\u00e9"}',
]


def verify(model_dir, path, n_base):
    import tiktoken
    from tokenizers import Tokenizer

    ranks = load_tiktoken(os.path.join(model_dir, "tiktoken.model"))
    added = special_tokens(model_dir, n_base)
    ref = tiktoken.Encoding(name="k3", pat_str=PAT, mergeable_ranks=ranks,
                            special_tokens={a["content"]: a["id"] for a in added})
    got = Tokenizer.from_file(path)

    bad = 0
    for s in CORPUS:
        want = ref.encode(s, disallowed_special=())
        have = got.encode(s, add_special_tokens=False).ids
        if want != have:
            bad += 1
            print(f"MISMATCH on {s[:60]!r}\n  tiktoken {want[:24]}\n  tokenizer.json {have[:24]}")
            continue
        back = got.decode(have, skip_special_tokens=False)
        if back != s:
            bad += 1
            print(f"ROUND-TRIP FAIL on {s[:60]!r}\n  got {back[:60]!r}")
    # SPECIALS ARE VERIFIED AGAINST A DIFFERENT REFERENCE, deliberately.  tiktoken only emits a
    # special id when the caller passes `allowed_special`; with `disallowed_special=()` above it
    # encodes `<|im_end|>` as the 6 ordinary tokens of its literal text.  HF `added_tokens` always
    # extract.  The serving path is the one that matters and it wants extraction — the chat
    # template emits these markers and they must become ONE id each — so the reference for this
    # half is tiktoken with `allowed_special="all"`, which is what SGLang/vLLM also drive.
    for a in added[:16]:
        want = ref.encode(a["content"], allowed_special="all")
        ids = got.encode(a["content"], add_special_tokens=False).ids
        if ids != [a["id"]] or want != ids:
            bad += 1
            print(f"SPECIAL {a['content']!r} -> {ids}, want [{a['id']}] (tiktoken {want})")
    # ...and a special EMBEDDED in ordinary text still has to split cleanly on both sides.
    mixed = "before <|im_end|> after"
    if ref.encode(mixed, allowed_special="all") != got.encode(mixed, add_special_tokens=False).ids:
        bad += 1
        print(f"MISMATCH on embedded special {mixed!r}\n"
              f"  tiktoken {ref.encode(mixed, allowed_special='all')}\n"
              f"  tokenizer.json {got.encode(mixed, add_special_tokens=False).ids}")

    n = got.get_vocab_size(with_added_tokens=True)
    print(f"vocab {n} (base {n_base} + {len(added)} special)")
    if n != n_base + 256:
        bad += 1
        print(f"VOCAB SIZE {n} != {n_base + 256}")
    print(f"{len(CORPUS)} corpus strings, {bad} failure(s)")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Kimi-K3 snapshot dir (holds tiktoken.model)")
    ap.add_argument("--out", help="dir to write tokenizer.json into (default: --model)")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    out_dir = a.out or a.model
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokenizer.json")

    tok, n_base = build(a.model)
    with open(path, "w") as f:
        json.dump(tok, f, ensure_ascii=False)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB, "
          f"{len(tok['model']['vocab'])} vocab, {len(tok['model']['merges'])} merges)")

    if a.verify and not verify(a.model, path, n_base):
        sys.exit("VERIFY FAILED")


if __name__ == "__main__":
    main()
