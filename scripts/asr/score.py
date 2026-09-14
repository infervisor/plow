#!/usr/bin/env python3
"""Score complete ASR runs with explicit normalization and paired speaker bootstrap."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import unicodedata

import jiwer


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'").replace("‘", "'")
    text = "".join(" " if unicodedata.category(c).startswith("P") and c != "'" else c for c in text)
    return " ".join(text.split())


def counts(reference, hypothesis, characters=False):
    result = (jiwer.process_characters if characters else jiwer.process_words)(reference, hypothesis)
    return {"substitutions": result.substitutions, "deletions": result.deletions,
            "insertions": result.insertions, "reference_units": result.hits + result.substitutions + result.deletions}


def errors(row):
    return sum(row[k] for k in ["substitutions", "deletions", "insertions"])


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def aggregate(rows, key):
    total = {field: sum(row[key][field] for row in rows) for field in rows[0][key]}
    total["rate"] = errors(total) / total["reference_units"] if total["reference_units"] else None
    return total


def paired_interval(left, right, manifest, iterations=2000):
    groups = defaultdict(list)
    for utterance in left:
        groups[manifest[utterance]["speaker"]].append(utterance)
    speakers = sorted(groups)
    rng = random.Random(20260910)
    samples = []
    time_ratios = []
    for _ in range(iterations):
        ids = [u for speaker in rng.choices(speakers, k=len(speakers)) for u in groups[speaker]]
        units = sum(left[u]["normalized_words"]["reference_units"] for u in ids)
        delta = sum(errors(left[u]["normalized_words"]) - errors(right[u]["normalized_words"]) for u in ids)
        samples.append(delta / units)
        time_ratios.append(sum(left[u]["seconds"] for u in ids) / sum(right[u]["seconds"] for u in ids))
    return {"method": "paired speaker bootstrap", "replicates": iterations, "seed": 20260910,
            "normalized_wer_delta_95_interval": [percentile(samples, .025), percentile(samples, .975)],
            "inference_time_ratio_95_interval": [percentile(time_ratios, .025), percentile(time_ratios, .975)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    manifest = {row["id"]: row for row in manifest_rows if "excluded_reason" not in row}
    if len({row["id"] for row in manifest_rows}) != len(manifest_rows) or not manifest:
        raise ValueError("manifest has duplicate IDs or no eligible utterances")
    backends = defaultdict(dict)
    for path in args.results:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] != "result":
                continue
            utterance, backend = row["id"], row["backend"]
            if utterance not in manifest or utterance in backends[backend]:
                raise ValueError(f"unknown or duplicate result: {backend}/{utterance}")
            expected = manifest[utterance]
            if row["reference"] != expected["reference"] or abs(row["duration_seconds"] - expected["duration_seconds"]) > 1e-6:
                raise ValueError(f"manifest mismatch: {utterance}")
            if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
                raise ValueError(f"invalid timing: {utterance}")
            reference = expected["reference"]
            hypothesis = "" if "error" in row else row["text"]
            ref_normal, hyp_normal = normalize(reference), normalize(hypothesis)
            row.update(normalized_reference=ref_normal, normalized_hypothesis=hyp_normal,
                       raw_words=counts(reference, hypothesis),
                       normalized_words=counts(ref_normal, hyp_normal),
                       normalized_characters=counts("".join(ref_normal.split()), "".join(hyp_normal.split()), True))
            backends[backend][utterance] = row
    summaries = {}
    for backend, results in backends.items():
        if results.keys() != manifest.keys():
            raise ValueError(f"incomplete run: {backend}, {len(results)}/{len(manifest)} results")
        rows = list(results.values())
        times = [row["seconds"] for row in rows]
        summaries[backend] = {"utterances": len(rows), "failures": sum("error" in row for row in rows),
                              "audio_seconds": sum(row["duration_seconds"] for row in rows),
                              "inference_seconds": sum(times), "latency_p50_seconds": statistics.median(times),
                              "latency_p95_seconds": percentile(times, .95),
                              "aggregate_rtf": sum(times)/sum(row["duration_seconds"] for row in rows),
                              **{key: aggregate(rows, key) for key in ["raw_words", "normalized_words", "normalized_characters"]}}
    if not summaries:
        raise ValueError("no results")
    pairs = {}
    names = sorted(backends)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            pairs[f"{left} minus {right}"] = {
                "different_raw_transcripts": sum(backends[left][u]["text"] != backends[right][u]["text"] for u in manifest),
                "normalized_wer_delta": summaries[left]["normalized_words"]["rate"]-summaries[right]["normalized_words"]["rate"],
                "inference_time_ratio": summaries[left]["inference_seconds"]/summaries[right]["inference_seconds"],
                **paired_interval(backends[left], backends[right], manifest)}
    report = {"manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "result_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.results},
              "jiwer_version": importlib.metadata.version("jiwer"),
              "normalization": "NFKC, casefold, curly apostrophe mapping, punctuation to spaces except apostrophes, collapse whitespace; no number expansion; CER excludes whitespace",
              "selected": len(manifest_rows), "excluded": len(manifest_rows)-len(manifest),
              "summaries": summaries, "paired_comparisons": pairs,
              "rows": {backend: list(rows.values()) for backend, rows in backends.items()}}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "paired_comparisons": pairs}, indent=2))


if __name__ == "__main__":
    main()
