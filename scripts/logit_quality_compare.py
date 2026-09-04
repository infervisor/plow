#!/usr/bin/env python3
"""Compare dense-logit manifests on identical teacher-forced histories."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_manifest(path):
    data = json.loads(path.read_text())
    base = path.parent
    rows = {}
    repeated = {}
    for case in data["cases"]:
        item = dict(case)
        file = Path(item["file"])
        item["file"] = file if file.is_absolute() else base / file
        key = item["prompt_sha256_u32le"]
        if key in rows:
            repeated.setdefault(key, [rows[key]]).append(item)
        rows[key] = item
    return data, rows, repeated


def load_row(case):
    if case.get("dtype") == "bf16" or case["file"].suffix == ".bin":
        bits = np.fromfile(case["file"], dtype="<u2").astype(np.uint32) << 16
        return bits.view(np.float32)
    return np.fromfile(case["file"], dtype="<f4")


def metrics(a, b, topk):
    if len(a) != len(b):
        raise ValueError(f"vocabulary mismatch: candidate={len(a)}, reference={len(b)}")
    n = len(a)
    a, b = a.astype(np.float64), b.astype(np.float64)
    # Centering removes the arbitrary scalar offset between logits and logprobs.
    ac, bc = a - a.mean(), b - b.mean()
    delta = ac - bc
    order_a, order_b = np.argsort(a)[::-1], np.argsort(b)[::-1]
    head = order_b[: min(64, n)]
    full_rel = np.linalg.norm(delta) / max(np.linalg.norm(bc), 1e-30)
    head_rel = np.linalg.norm(delta[head]) / max(np.linalg.norm(bc[head]), 1e-30)
    overlap = {
        str(k): len(set(order_a[: min(k, n)]) & set(order_b[: min(k, n)])) / min(k, n)
        for k in topk
    }
    token_a, token_b = int(order_a[0]), int(order_b[0])
    gap_a, gap_b = float(a[order_a[0]] - a[order_a[1]]), float(b[order_b[0]] - b[order_b[1]])
    max_abs = float(np.max(np.abs(delta)))
    if token_a == token_b:
        classification = "same-token"
    elif min(gap_a, gap_b) <= max_abs:
        classification = "near-tie-flip"
    else:
        classification = "gap-exceeds-max-error"
    return {
        "vocab_compared": n,
        "full_row_centered_rel_l2": float(full_rel),
        "reference_head64_centered_rel_l2": float(head_rel),
        "centered_max_abs": max_abs,
        "topk_overlap": overlap,
        "token": {"candidate": token_a, "reference": token_b},
        "gap": {"candidate": gap_a, "reference": gap_b},
        "classification": classification,
    }


def repeat_checks(meta, repeated, topk):
    checks = list(meta.get("repeat_checks") or [])
    if checks:
        return checks
    for key, cases in repeated.items():
        first = cases[0]
        for case in cases[1:]:
            row = metrics(load_row(case), load_row(first), topk)
            checks.append(
                {
                    "first_case": first["id"],
                    "repeat_case": case["id"],
                    "prompt_sha256_u32le": key,
                    "full_row_centered_rel_l2": row["full_row_centered_rel_l2"],
                    "reference_head64_centered_rel_l2": row[
                        "reference_head64_centered_rel_l2"
                    ],
                    "centered_max_abs": row["centered_max_abs"],
                    "top64_overlap": row["topk_overlap"].get("64", 1.0),
                    "same_argmax": row["token"]["candidate"]
                    == row["token"]["reference"],
                }
            )
    return checks


def repeat_floor(checks, sources):
    if not checks:
        return None
    return {
        "scope": "global-conservative-maximum",
        "sources": sources,
        "repeat_pairs": len(checks),
        "full_row_centered_rel_l2": max(
            x["full_row_centered_rel_l2"] for x in checks
        ),
        "reference_head64_centered_rel_l2": max(
            x["reference_head64_centered_rel_l2"] for x in checks
        ),
        "centered_max_abs": max(x["centered_max_abs"] for x in checks),
        "minimum_top64_overlap": min(x["top64_overlap"] for x in checks),
        "argmax_flips": sum(not x["same_argmax"] for x in checks),
    }
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--candidate", required=True, action="append", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--top-k", default="1,5,16,64")
    p.add_argument("--repeat-floor-manifest", action="append", type=Path, default=[])
    p.add_argument("--repeat-floor-multiplier", type=float, default=2.0)
    args = p.parse_args()
    topk = [int(x) for x in args.top_k.split(",")]
    ref_meta, refs, ref_repeated = load_manifest(args.reference)
    checks = repeat_checks(ref_meta, ref_repeated, topk)
    floor_sources = [str(args.reference)] if checks else []
    for path in args.repeat_floor_manifest:
        floor_meta, _, floor_repeated = load_manifest(path)
        extra = repeat_checks(floor_meta, floor_repeated, topk)
        if extra:
            checks.extend(extra)
            floor_sources.append(str(path))
    floor = repeat_floor(checks, floor_sources)
    unstable_histories = {
        x["prompt_sha256_u32le"] for x in checks if not x["same_argmax"]
    }
    repeated_histories = {x["prompt_sha256_u32le"] for x in checks}
    report = {
        "schema": 2,
        "reference": str(args.reference),
        "reference_repeat_floor": floor,
        "repeat_floor_multiplier": args.repeat_floor_multiplier,
        "comparisons": [],
    }
    lines = ["# Dense-logit quality comparison", ""]
    if floor:
        lines += [
            "Reference repeat floor (conservative maximum): "
            f"full={floor['full_row_centered_rel_l2']:.6g}, "
            f"head64={floor['reference_head64_centered_rel_l2']:.6g}; "
            f"acceptance multiplier={args.repeat_floor_multiplier:g}x.",
            "",
        ]
    else:
        lines += ["No reference repeats supplied; no quality verdict is emitted.", ""]
    for candidate_path in args.candidate:
        cand_meta, candidates, _ = load_manifest(candidate_path)
        name = cand_meta.get("name", candidate_path.stem)
        rows = []
        for key, cand in candidates.items():
            ref = refs.get(key)
            if ref is None:
                continue
            row = metrics(load_row(cand), load_row(ref), topk)
            row.update(
                {
                    "candidate_case": cand["id"],
                    "reference_case": ref["id"],
                    "prompt_len": cand["prompt_len"],
                }
            )
            row["prompt_sha256_u32le"] = key
            row["reference_argmax_unstable"] = key in unstable_histories
            row["reference_repeat_status"] = (
                "unstable"
                if key in unstable_histories
                else "observed-stable"
                if key in repeated_histories
                else "unmeasured"
            )
            if row["reference_argmax_unstable"] and row["classification"] != "same-token":
                row["classification"] = "reference-repeat-unstable"
            if floor:
                full_ratio = row["full_row_centered_rel_l2"] / max(
                    floor["full_row_centered_rel_l2"], 1e-30
                )
                head_ratio = row["reference_head64_centered_rel_l2"] / max(
                    floor["reference_head64_centered_rel_l2"], 1e-30
                )
                row["repeat_floor_ratio"] = {
                    "full_row": full_ratio,
                    "reference_head64": head_ratio,
                }
                row["outside_repeat_floor"] = (
                    full_ratio > args.repeat_floor_multiplier
                    or head_ratio > args.repeat_floor_multiplier
                )
            rows.append(row)
        if not rows:
            raise RuntimeError(f"{name}: no matching teacher-forced histories")
        severe = sum(r["classification"] == "gap-exceeds-max-error" for r in rows)
        outside = sum(r.get("outside_repeat_floor", False) for r in rows)
        token_agreement = sum(
            r["token"]["candidate"] == r["token"]["reference"] for r in rows
        )
        summary = {
            "name": name,
            "matched_histories": len(rows),
            "gap_exceeds_row_error_flips": severe,
            "token_agreement_rows": token_agreement,
            "reference_unstable_rows": sum(
                r["reference_repeat_status"] == "unstable" for r in rows
            ),
            "reference_repeat_unmeasured_rows": sum(
                r["reference_repeat_status"] == "unmeasured" for r in rows
            ),
            "rows_outside_repeat_floor": outside if floor else None,
            "quality_gate_scope": "all-matched-exact-teacher-forced-histories",
            "quality_gate_pass": (outside == 0) if floor else None,
            "longest_prompt_tokens": max(r["prompt_len"] for r in rows),
            "median_full_row_centered_rel_l2": float(
                np.median([r["full_row_centered_rel_l2"] for r in rows])
            ),
            "median_reference_head64_centered_rel_l2": float(
                np.median(
                    [r["reference_head64_centered_rel_l2"] for r in rows]
                )
            ),
            "minimum_top64_overlap": float(min(r["topk_overlap"].get("64", 1.0) for r in rows)),
            "rows": rows,
        }
        report["comparisons"].append(summary)
        verdict = (
            f"rows outside {args.repeat_floor_multiplier:g}x repeat floor: {outside}; "
            f"gate: {'pass' if summary['quality_gate_pass'] else 'fail'}"
            if floor
            else "gate: unavailable"
        )
        lines += [
            f"## {name}",
            "",
            f"Matched exact-history rows: {len(rows)} through prompt length "
            f"{summary['longest_prompt_tokens']}; token agreement: "
            f"{token_agreement}/{len(rows)}; {verdict}.",
            "",
            "| prompt | full relL2 | head64 relL2 | floor ratio full/head | "
            "top64 | token | class |",
            "|---:|---:|---:|---:|---:|---|---|",
        ]
        for row in rows:
            ratios = row.get("repeat_floor_ratio")
            ratio_text = (
                f"{ratios['full_row']:.2f} / {ratios['reference_head64']:.2f}"
                if ratios
                else "n/a"
            )
            lines.append(
                f"| {row['prompt_len']} | {row['full_row_centered_rel_l2']:.6g} | "
                f"{row['reference_head64_centered_rel_l2']:.6g} | {ratio_text} | "
                f"{row['topk_overlap'].get('64', 1.0):.3f} | "
                f"{row['token']['candidate']} / {row['token']['reference']} | "
                f"{row['classification']} |"
            )
        lines.append("")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
