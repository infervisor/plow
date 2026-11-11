#!/usr/bin/env python3
"""CPU speech-tag inventory from a tokenizer and packet text rules. Prints an advisory map.

Use --verify-ids with the existing tokenizer reference Python to retain actual encoding IDs.
Unsupported text rules are reported as incomplete; no GPU or model weights are loaded.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recipe_common as rc


def aliases() -> dict[str, str]:
    source = (Path(__file__).resolve().parents[2] / "crates/plowrt/src/tts/tags.rs").read_text()
    function = source.split("fn canonical(s:", 1)[1].split("#[derive", 1)[0]
    return {alias: canonical for left, canonical in re.findall(r'([^\n]+) => "([a-z_]+)",', function)
            for alias in re.findall(r'"([^"]+)"', left)}


def text_rules(text: str, strings: dict, language: str | None, tables=None, trusted=True) -> tuple[str, list[str]]:
    unsupported = []
    tables = tables or {}

    def run(value, rules):
        for line in rules.splitlines():
            if not line:
                continue
            kind, *args = line.split("\t")
            args = [s.replace("{lang}", language or "") for s in args]
            if kind == "default_if_empty":
                value = value or args[0]
            elif kind == "capitalize_first":
                if value:
                    value = value[0].upper() + value[1:]
            elif kind == "collapse_whitespace":
                value = " ".join(value.split())
            elif kind == "replace":
                value = value.replace(*args)
            elif kind == "trim_end":
                value = value.rstrip(args[0])
            elif kind == "ensure_suffix":
                if not value.endswith(tuple(args[0])):
                    value += args[1]
            elif kind == "lowercase":
                value = value.lower()
            elif kind == "nfkd":
                value = unicodedata.normalize("NFKD", value)
            elif kind == "strip":
                value = value.strip()
            elif kind == "prefix":
                if trusted:
                    value = args[0] + value
            elif kind == "language":
                value = run(value, strings.get("text.rules.lang." + (language or ""), ""))
            elif kind in ("map_chars", "dict_longest") and args[0] in tables:
                pairs = dict(row.split("\t", 1) if "\t" in row else (row, "")
                             for row in tables[args[0]].splitlines() if row)
                if kind == "map_chars":
                    value = "".join(pairs.get(ch, ch) for ch in value)
                elif pairs:
                    value = re.sub("|".join(re.escape(k) for k in sorted(pairs, key=len, reverse=True)),
                                   lambda m: pairs[m[0]], value)
            elif kind == "drop_chars":
                ranges = [tuple(int(n, 16) for n in (part.split("-") if "-" in part else [part, part]))
                          for part in args[0].split(",") if part]
                value = "".join(c for c in value if not any(lo <= ord(c) <= hi for lo, hi in ranges))
            else:
                unsupported.append(line)
        return value

    return run(text, strings.get("text.rules", "")), sorted(set(unsupported))


def token_inventory(tokenizer, strings):
    alias = aliases()
    langs = {k.removeprefix("text.rules.lang.") for k in strings if k.startswith("text.rules.lang.")}
    vocabulary = tokenizer.get("model", {}).get("vocab", {})
    if isinstance(vocabulary, list):
        vocabulary = {v[0]: i for i, v in enumerate(vocabulary)}
    tokens = {content: dict(token=content, id=token_id, added=False, special=False)
              for content, token_id in vocabulary.items()}
    for entry in tokenizer.get("added_tokens", []):
        tokens[entry["content"]] = dict(token=entry["content"], id=entry["id"], added=True,
                                        special=entry.get("special", False))
    found = []
    for content, row in tokens.items():
        inner = content.strip("[]<>()|/ ").lower()
        canonical = alias.get(inner)
        if canonical and (row["added"] or re.fullmatch(r"[<\[(].*[>\])]", content)):
            row.update(kind="event" if canonical in rc.CANONICAL_TAGS[:12] else "style", canonical=canonical)
        elif inner in langs or re.fullmatch(r"\[[a-z]{2}(?:-[a-z]+)?\]", content):
            row.update(kind="language", canonical=None)
        elif row["special"] or re.fullmatch(r"(?:<[^>]+>|\[[^\]]+\])", content):
            row.update(kind="control", canonical=None)
        else:
            continue
        row["accepted_spellings"] = ([form for word, canonical_name in alias.items() if canonical_name == canonical
                                      for form in (f"[{word}]", f"<{word}>", f"*{word}*")]
                                     if canonical else [])
        found.append(row)
    return sorted(found, key=lambda r: (r["id"], r["token"]))


def compatibility(text):
    return "".join(chr(ord(c) + 0xFEE0) if 0x21 <= ord(c) <= 0x7E else c for c in text)


def discover(tokenizer, pipelines, encode=None):
    proposed = rc.default_speech_tag_map()
    reports = []
    all_unsupported = set()
    for pipeline in pipelines:
        strings = pipeline.get("strings", {})
        languages = [k.removeprefix("text.rules.lang.") for k in strings if k.startswith("text.rules.lang.")]
        languages = languages or [strings.get("text.default_language")]
        inventory = token_inventory(tokenizer, strings)
        hazards = []
        # Each candidate is independent: trusted language prefixes cannot masquerade as injection.
        for row in inventory:
            token = row["token"]
            forms = [("token", token), ("compatibility", compatibility(token)),
                     ("compatibility", token.translate(str.maketrans("[]<>*", "［］＜＞＊")))]
            if token.startswith("["):
                forms += [("escaped", "\\" + token), ("nested", "[" + token + "]"),
                          ("uppercase", token.upper())]
            for language in languages:
                for form, text in forms:
                    normalized, unknown = text_rules(text, strings, language, trusted=False)
                    all_unsupported.update(unknown)
                    trusted_text, _ = text_rules(text, strings, language)
                    clean_text, _ = text_rules("Hello.", strings, language)
                    ids = list(encode(trusted_text)) if encode and not unknown else None
                    baseline_ids = list(encode(clean_text)) if encode and not unknown else None
                    injected = (ids.count(row["id"]) - baseline_ids.count(row["id"])) if ids is not None else None
                    if token not in normalized and not (injected is not None and injected > 0):
                        continue
                    fullwidth = form == "compatibility" and text != token
                    hazards.append(dict(token=token, token_id=row["id"], kind=row["kind"], canonical=row["canonical"],
                                        form=form, input=text, language=language, normalized=normalized,
                                        accepted_syntax=("marker" if text.startswith("<|") else
                                                         "bracket" if text.startswith("[") and form not in ("nested",) else
                                                         "ssml" if text.startswith("<") else "literal"),
                                        ascii_trigger=any(c in text for c in "[<*\\"),
                                        compatibility_bypasses_parser=fullwidth and not any(c in text for c in "[<*\\"),
                                        evidence="encoded" if ids is not None else "added-token string collision" if row["added"] else "vocabulary candidate; encode to confirm",
                                        ids=ids, baseline_ids=baseline_ids,
                                        injected_count=injected,
                                        rules_complete=not unknown))
        if any(h["token"].startswith("[") for h in hazards):
            proposed["literal_rewrite"] = [["<|", "< |"], ["[", "("], ["]", ")"]]
        reports.append(dict(pipeline=pipeline["name"], driver=pipeline["driver"], inventory=inventory, collisions=hazards))
    return dict(version=1, scope="Advisory discovery, not a safe-input or native-efficacy certificate",
                pipelines=reports, unsupported_rules=sorted(all_unsupported), complete=not all_unsupported,
                proposed_policy=dict(version=1, profile="strip-silence-v1", map=proposed),
                literal_rewrites_needed=proposed["literal_rewrite"],
                unicode_scope="Compatibility forms can become control tokens after normalization while bypassing the ASCII tag parser. ASCII literal rewrites do not protect them; an explicit frontend/owner decision is required.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--verify-ids", action="store_true", help="encode with the existing reference Python's tokenizers package (CPU)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    encode = None
    if a.verify_ids:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(a.tokenizer))
        encode = lambda text: tok.encode(text, add_special_tokens=False).ids
    pipelines = rc.speech_pipelines(a.packet)
    if not pipelines:
        ap.error("packet has no speech pipeline")
    result = discover(rc.load_json(a.tokenizer), pipelines, encode)
    result.update(packet_sha256=rc.sha256_file(a.packet), tokenizer_sha256=rc.sha256_file(a.tokenizer))
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if a.out:
        rc.write_atomic(a.out, output.encode())
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
