#!/usr/bin/env python3
"""Q0 oracle plumbing for a two-arm Plow-vs-vLLM logit comparison.

Subcommands, in gate order (`docs/k3-mi355x-20260904/scripts/mla_materialized_gate.sh` drives them):

  prompts  build the fixed prompt set (token-id rows at exact lengths) from GSM8K text;
           needs a tokenizer, so it runs inside the pinned vLLM container
  merge    fold the per-prompt `plow_logit_manifest.py` outputs of one arm into one manifest
  tokens   per prompt, the first greedy-token divergence between two arms' manifests
  cases    the union of both arms' teacher-forced histories as `vllm_logit_oracle.py` cases,
           each listed twice so the oracle measures its own repeat floor
  verdict  apply the acceptance rule to a `logit_quality_compare.py` report:
           top-1 agreement >= 99.5% of matched histories, full-row centered relL2 within
           `--floor-multiplier` (2x) of the vLLM repeat floor on >= `--within-fraction` of
           rows, and no `gap-exceeds-max-error` flip

Only `prompts` imports a tokenizer; everything else is stdlib.
"""

import argparse
import array
import hashlib
import json
import sys
from pathlib import Path


def digest(ids):
    values = array.array("I", ids)
    if values.itemsize != 4:
        raise RuntimeError("host unsigned int is not 32 bits")
    if sys.byteorder != "little":
        values.byteswap()
    return hashlib.sha256(values.tobytes()).hexdigest()


def load_tokenizer(path):
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return lambda text: [int(x) for x in tok.encode(text)]
    except ImportError:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        return lambda text: [int(x) for x in tok.encode(text).ids]


def cmd_prompts(args):
    encode = load_tokenizer(args.tokenizer)
    questions = [json.loads(line)["question"] for line in open(args.gsm8k) if line.strip()]
    if not questions:
        raise ValueError(f"{args.gsm8k} has no questions")
    lengths = [int(x) for x in args.lengths.split(",")]
    prompts = []
    for index, length in enumerate(lengths):
        # A different question window per length so no prompt is a prefix of another; the
        # text is joined until it covers `length` tokens and cut to exactly that many.
        start = (index * 37) % len(questions)
        parts = []
        ids = []
        cursor = start
        while len(ids) < length:
            parts.append(questions[cursor % len(questions)])
            cursor += 1
            ids = encode("\n\n".join(parts))
            if cursor - start > 4 * len(questions):
                raise RuntimeError(f"cannot reach {length} tokens from GSM8K text")
        prompts.append({"id": f"gsm{length}", "len": length, "ids": ids[:length]})
    out = {
        "schema": 1,
        "source": "gsm8k-test-questions",
        "tokenizer": str(args.tokenizer),
        "prompts": prompts,
    }
    args.output.write_text(json.dumps(out) + "\n")
    print(f"wrote {len(prompts)} prompts: " + ", ".join(f"{p['id']}={p['len']}" for p in prompts))


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    base = Path(path).parent
    for case in data["cases"]:
        file = Path(case["file"])
        case["file"] = str(file if file.is_absolute() else base / file)
    return data


def cmd_merge(args):
    cases = []
    for path in args.manifest:
        data = load_manifest(path)
        for step, case in enumerate(data["cases"]):
            case = dict(case)
            case["prompt_id"] = data["name"].split("-", 1)[1] if "-" in data["name"] else data["name"]
            case["step"] = step
            cases.append(case)
    out = {"schema": 1, "producer": "plow-amd-bench", "name": args.name, "cases": cases}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{args.name}: {len(cases)} teacher-forced histories from {len(args.manifest)} prompts")


def by_prompt(data):
    groups = {}
    for case in data["cases"]:
        groups.setdefault(case["prompt_id"], []).append(case)
    for cases in groups.values():
        cases.sort(key=lambda c: c["step"])
    return groups


def cmd_tokens(args):
    a, b = load_manifest(args.left), load_manifest(args.right)
    ga, gb = by_prompt(a), by_prompt(b)
    print(f"{'prompt':<10} {'len':>6} {'steps':>5}  first divergence  ({a['name']} vs {b['name']})")
    exact = True
    for pid in sorted(set(ga) | set(gb), key=lambda p: ga.get(p, gb.get(p))[0]["prompt_len"]):
        if pid not in ga or pid not in gb:
            print(f"{pid:<10} missing in one arm")
            exact = False
            continue
        ta = [c["sampled_token_id"] for c in ga[pid]]
        tb = [c["sampled_token_id"] for c in gb[pid]]
        n = min(len(ta), len(tb))
        diverge = next((i for i in range(n) if ta[i] != tb[i]), None)
        length = ga[pid][0]["prompt_len"]
        if diverge is None:
            print(f"{pid:<10} {length:>6} {n:>5}  identical")
        else:
            exact = False
            print(
                f"{pid:<10} {length:>6} {n:>5}  step {diverge}: {ta[diverge]} vs {tb[diverge]}"
                f"  ({sum(x != y for x, y in zip(ta, tb))}/{n} differ)"
            )
    print("TOKENS_IDENTICAL" if exact else "TOKENS_DIFFER (1-ULP class expected; decide on the logit oracle)")


def cmd_cases(args):
    seen = {}
    for path in args.manifest:
        for case in load_manifest(path)["cases"]:
            sha = digest(case["prompt_token_ids"])
            seen.setdefault(sha, case)
    cases = []
    for sha, case in seen.items():
        for repeat in range(args.repeats):
            cid = case["id"] if repeat == 0 else f"{case['id']}-r{repeat + 1}"
            cases.append({"id": cid, "prompt_token_ids": case["prompt_token_ids"]})
    args.output.write_text(json.dumps({"cases": cases}) + "\n")
    print(f"{len(seen)} distinct histories x {args.repeats} repeats = {len(cases)} oracle cases")


def read_bf16_row(path):
    raw = array.array("H")
    with open(path, "rb") as f:
        raw.frombytes(f.read())
    if sys.byteorder != "little":
        raw.byteswap()
    bits = array.array("I", (x << 16 for x in raw))
    return array.array("f", bits.tobytes())


def cmd_diff(args):
    """Two `amd-bench --dump-logits` directories of the same prompt, step by step."""
    import math

    tags = ["prefill"] + [f"{i:03}" for i in range(args.steps)]
    print(f"{'step':<8} {'argmax':>14}  {'centered relL2':>14}  {'max|d|':>10}  {'top-1 gap L/R':>16}")
    for tag in tags:
        a_path, b_path = args.left / f"logits_{tag}.bin", args.right / f"logits_{tag}.bin"
        if not a_path.is_file() or not b_path.is_file():
            break
        a, b = read_bf16_row(a_path), read_bf16_row(b_path)
        if len(a) != len(b):
            raise ValueError(f"{tag}: vocab {len(a)} vs {len(b)}")
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = den = 0.0
        max_abs = 0.0
        for x, y in zip(a, b):
            xc, yc = x - ma, y - mb
            num += (xc - yc) ** 2
            den += yc**2
            max_abs = max(max_abs, abs(xc - yc))
        ia = max(range(len(a)), key=a.__getitem__)
        ib = max(range(len(b)), key=b.__getitem__)
        gap = lambda v, i: v[i] - max(x for j, x in enumerate(v) if j != i)
        print(
            f"{tag:<8} {ia:>6} {ib:>6}{'  ' if ia == ib else ' *'} {math.sqrt(num / max(den, 1e-30)):>14.3e}"
            f"  {max_abs:>10.4f}  {gap(a, ia):>7.3f} {gap(b, ib):>7.3f}"
        )


def cmd_verdict(args):
    report = json.loads(args.report.read_text())
    floor = report.get("reference_repeat_floor")
    if not floor:
        print("FAIL: the reference manifest carries no repeat floor (run the oracle with repeats)")
        sys.exit(2)
    multiplier = report.get("repeat_floor_multiplier", 2.0)
    print(
        f"vLLM repeat floor: full-row relL2 {floor['full_row_centered_rel_l2']:.4g}, "
        f"head64 {floor['reference_head64_centered_rel_l2']:.4g}, "
        f"argmax flips {floor['argmax_flips']}/{floor['repeat_pairs']} pairs; "
        f"acceptance = {multiplier:g}x floor"
    )
    ok_all = True
    for cand in report["comparisons"]:
        rows = cand["rows"]
        matched = len(rows)
        agree = cand["token_agreement_rows"] / matched
        within = sum(
            r["repeat_floor_ratio"]["full_row"] <= multiplier for r in rows
        ) / matched
        severe = cand["gap_exceeds_row_error_flips"]
        worst = max(r["repeat_floor_ratio"]["full_row"] for r in rows)
        median = cand["median_full_row_centered_rel_l2"] / max(
            floor["full_row_centered_rel_l2"], 1e-30
        )
        ok = agree >= args.top1 and within >= args.within_fraction and severe == 0
        ok_all &= ok
        print(
            f"{cand['name']:<24} rows {matched:>4}  top-1 {agree:7.3%}  "
            f"within {multiplier:g}x floor {within:7.3%} (median {median:.2f}x, worst {worst:.2f}x)  "
            f"severe flips {severe}  -> {'PASS' if ok else 'FAIL'}"
        )
    print("Q0_PASS" if ok_all else "Q0_FAIL")
    sys.exit(0 if ok_all else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prompts")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--gsm8k", required=True, type=Path)
    s.add_argument("--lengths", default="300,1024,8192,8400,9000")
    s.add_argument("--output", required=True, type=Path)
    s.set_defaults(run=cmd_prompts)

    s = sub.add_parser("merge")
    s.add_argument("--name", required=True)
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("manifest", nargs="+", type=Path)
    s.set_defaults(run=cmd_merge)

    s = sub.add_parser("tokens")
    s.add_argument("--left", required=True, type=Path)
    s.add_argument("--right", required=True, type=Path)
    s.set_defaults(run=cmd_tokens)

    s = sub.add_parser("cases")
    s.add_argument("--output", required=True, type=Path)
    s.add_argument("--repeats", type=int, default=2)
    s.add_argument("manifest", nargs="+", type=Path)
    s.set_defaults(run=cmd_cases)

    s = sub.add_parser("diff")
    s.add_argument("--left", required=True, type=Path)
    s.add_argument("--right", required=True, type=Path)
    s.add_argument("--steps", type=int, default=16)
    s.set_defaults(run=cmd_diff)

    s = sub.add_parser("verdict")
    s.add_argument("--report", required=True, type=Path)
    s.add_argument("--top1", type=float, default=0.995)
    s.add_argument("--within-fraction", type=float, default=0.9)
    s.set_defaults(run=cmd_verdict)

    args = p.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
