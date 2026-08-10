#!/usr/bin/env python3
"""FACTS GATE — does a prefill chunk-plan change DEGRADE answers, or merely reword them?

Every acceptance instrument plow has today is CHARACTER IDENTITY, which fires on
harmless reassociation (57.8% of prompt lengths reword under `PLOW_RAGGED_CHUNK`)
and says nothing about quality. This is the other instrument: a battery of
prompts whose answers are MACHINE-CHECKABLE, run at OFF-RUNG lengths, compared
PAIRWISE between two arms.

Three design constraints, each from a measurement in
`perf-data/plow-gfx942/glm52-chunk-policy.md`:

  * **The answer must be LATE.** Divergence between two chunk plans lands at
    median character ~100 of a ~1150-character answer (~11% in). A needle in the
    first sentence is on the identical side of the divergence and cannot fail.
    Every item here forces the reasoning FIRST and the checkable token onto the
    LAST line, and `answer_frac` records how far in it landed so a battery that
    stops being late fails loudly instead of silently losing power.
  * **The battery must be able to fail.** 60/60 verifiable answers were already
    correct across all three arms on the 5-question predecessor; a small battery
    passes trivially and proves nothing. This one is 12 items x 7 lengths = 84
    cells, weighted toward multi-hop chains (one perturbation propagates),
    retrieval at depth (the needle sits IN the prefill, where a chunk plan can
    lose it), and structured output (a dropped field is detectable).
  * **It must run OFF-RUNG.** On-rung lengths are byte-identical across arms and
    carry no signal at all. The default `--lens` are the seven lengths where the
    ctrl and ragged plans differ.

The verdict is PAIRED (McNemar), not absolute: "is the candidate worse than the
baseline", not "is the candidate good". A cell both arms get wrong is a model
limit, not a regression. And a baseline that cannot itself answer the battery
makes the comparison powerless, so a weak baseline is an ERROR (exit 2), never a
PASS — that is the `PLOW_FUSE_QUANT` failure mode, where a gate stayed green for
weeks because it had no power to be red.

    run      one arm against a served port    -> JSON
    verdict  candidate JSON vs baseline JSON  -> table, exit 0/1/2

`run --inject` corrupts the prompt the way a chunk-plan bug corrupts it — same
token count (so the PLAN is unchanged), content replaced with filler — which is
how this gate was demonstrated to fail before it was trusted to pass.
"""
import argparse, json, math, re, sys, time, urllib.request
from collections import defaultdict

FILLER = " the"  # exactly one token under the GLM tokenizer; verified per cell

# Reason first, answer last. The whole point is to put the checkable token PAST
# the character at which two chunk plans separate.
FMT_ANSWER = (
    "\n\nWork through this carefully. Explain your reasoning in four to six full "
    "sentences BEFORE you give any answer, and do not state the answer early. "
    "Then finish your reply with a final line of exactly this form, with nothing "
    "after it:\nANSWER: <value>"
)
FMT_FIELDS = (
    "\n\nExplain your method in four to six full sentences BEFORE giving any "
    "values. Then end your reply with the result lines, one per line, each in "
    "exactly the form KEY=VALUE, with nothing after them."
)

HEAD = "Read the following record, then answer the question that comes after it.\n\n"


def _hop(iid, q, expect):
    return {"id": iid, "cls": "hop", "head": HEAD, "needles": [], "q": q,
            "fmt": FMT_ANSWER, "kind": "exact", "expect": expect}


# --- the battery ------------------------------------------------------------
# 12 items. Ground truth for every arithmetic chain is computed independently in
# `selftest`, so a mistyped expectation fails the harness rather than an arm.
ITEMS = [
    # (a) MULTI-HOP ARITHMETIC — a single perturbation anywhere in the chain
    # changes the final token, and the final token is the only thing checked.
    _hop("hop_train",
         "A freight train departs at 06:40. It travels for 2 hours and 35 minutes, "
         "then stops for 40 minutes, then travels for a further 3 hours and 55 "
         "minutes. At what time does it arrive? Use a 24-hour clock.",
         "13:50"),
    _hop("hop_tank",
         "A tank holds 3 cubic metres, and 1 cubic metre is 1000 litres. The tank "
         "is 40 percent full. Then 480 litres are added. Then 15 percent of the "
         "resulting volume is drained away. How many litres remain?",
         "1428"),
    _hop("hop_money",
         "An item costs 240 pounds. Its price first rises by 25 percent, then "
         "falls by 20 percent from that raised price, and finally a flat discount "
         "of 45 pounds is applied. What is the final price in pounds?",
         "195"),
    _hop("hop_mod",
         "Compute 7 raised to the power 4, then subtract 401 from it, then take "
         "the remainder of that result when divided by 37. What is the remainder?",
         "2"),
    _hop("hop_avg",
         "Five sensor readings are 12, 19, 7, 24 and 13. Compute their mean, "
         "multiply the mean by 8, then subtract the median of the five readings "
         "from that product. What is the result?",
         "107"),

    # (b) RETRIEVAL AT DEPTH — the needle sits inside the PREFILL, at a
    # controlled token depth, which is exactly the region a chunk-plan change
    # re-partitions. `late` sits in the final chunk; `mid` straddles the interior.
    {"id": "ndl_early", "cls": "needle", "head": HEAD,
     "needles": [(0.25, " Maintenance note: the access code for vault ALPHA is QF7392.")],
     "q": "What is the access code for vault ALPHA?",
     "fmt": FMT_ANSWER, "kind": "exact", "expect": "QF7392"},
    {"id": "ndl_mid", "cls": "needle", "head": HEAD,
     "needles": [(0.55, " Maintenance note: the access code for vault BRAVO is MX4185.")],
     "q": "What is the access code for vault BRAVO?",
     "fmt": FMT_ANSWER, "kind": "exact", "expect": "MX4185"},
    {"id": "ndl_late", "cls": "needle", "head": HEAD,
     "needles": [(0.94, " Maintenance note: the access code for vault CHARLIE is ZD6031.")],
     "q": "What is the access code for vault CHARLIE?",
     "fmt": FMT_ANSWER, "kind": "exact", "expect": "ZD6031"},
    # Retrieval BY KEY, not by position: the wanted code is the middle one, so an
    # arm that has only kept the most recent record answers confidently and wrong.
    {"id": "ndl_key", "cls": "needle", "head": HEAD,
     "needles": [(0.20, " Record: unit DELTA was assigned serial 5521."),
                 (0.50, " Record: unit ECHO was assigned serial 8074."),
                 (0.80, " Record: unit FOXTROT was assigned serial 3160.")],
     "q": "Which serial was assigned to unit ECHO?",
     "fmt": FMT_ANSWER, "kind": "exact", "expect": "8074"},
    # Multi-hop ACROSS depths: needs both ends of the context, so losing either
    # one is fatal and neither can be guessed.
    {"id": "ndl_sum", "cls": "needle", "head": HEAD,
     "needles": [(0.30, " Ledger entry: account GOLF holds 4817 units."),
                 (0.90, " Ledger entry: account HOTEL holds 2694 units.")],
     "q": "What is the combined total held by accounts GOLF and HOTEL?",
     "fmt": FMT_ANSWER, "kind": "exact", "expect": "7511"},

    # (c) STRUCTURED OUTPUT — every field is checked, so a DROPPED field is
    # detectable. Free-form graders cannot see an omission.
    {"id": "str_num", "cls": "struct", "head": HEAD, "needles": [],
     "q": "Consider the number 1848. Report four values about it: DOUBLE (the "
          "number doubled), HALF (the number halved), DIGITSUM (the sum of its "
          "digits) and REVERSED (its digits written in reverse order).",
     "fmt": FMT_FIELDS, "kind": "fields",
     "expect": [["DOUBLE", "3696"], ["HALF", "924"], ["DIGITSUM", "21"], ["REVERSED", "8481"]]},
    {"id": "str_sq", "cls": "struct", "head": HEAD, "needles": [],
     "q": "Report the squares of the six integers from 12 to 17 inclusive, as six "
          "lines with the keys SQ12, SQ13, SQ14, SQ15, SQ16 and SQ17.",
     "fmt": FMT_FIELDS, "kind": "fields",
     "expect": [["SQ12", "144"], ["SQ13", "169"], ["SQ14", "196"],
                ["SQ15", "225"], ["SQ16", "256"], ["SQ17", "289"]]},
]
BY_ID = {it["id"]: it for it in ITEMS}

# The seven lengths at which the ctrl and ragged chunk plans DIFFER. On-rung
# lengths (1024, 4096, 8192, ...) are byte-identical across arms by construction
# and would only dilute the battery with cells that cannot carry signal.
DEFAULT_LENS = "1025,3073,4097,6145,8193,10369,12345"
# The compiled GLM-5.2 prefill ladder (8k form), used ONLY to refuse cells that
# cannot carry signal.
LADDER = (128, 512, 1024, 2048, 4096, 8192)
LAUNCH_ROWS = 416   # mirrors exec::amd::LAUNCH_ROWS


def cover_ragged(n, bkt=LADDER):
    """Mirror of `plan_chunks_cfg(.., ragged=true)`: fewest launches."""
    out, rem, mx = [], n, max(bkt)
    while rem > mx:
        out.append(mx)
        rem -= mx
    if rem:
        out.append(min(b for b in bkt if b >= rem))
    return out


def cover_padded(n, bkt=LADDER, lr=LAUNCH_ROWS):
    """Mirror of the shipped padding-vs-launch DP."""
    q = min(bkt)
    rows = -(-n // q)
    cost = [0] + [float("inf")] * rows
    pick = [0] * (rows + 1)
    for r in range(1, rows + 1):
        for b in bkt:
            p = max(r - max(b // q, 1), 0)
            if cost[p] + b + lr < cost[r]:
                cost[r], pick[r] = cost[p] + b + lr, b
    out, r = [], rows
    while r > 0:
        out.append(pick[r])
        r = max(r - max(pick[r] // q, 1), 0)
    return sorted(out, reverse=True)


def carries_no_signal(n, bkt=LADDER):
    """True when both arms run the SAME plan AND the same last-chunk row count.

    The determinant of the output is the LAST chunk's EXECUTED row count
    (`glm52-chunk-policy.md` §2.2), so a length where that is equal in both arms
    is byte-identical by construction and can contribute nothing.
    """
    rg, pd = cover_ragged(n, bkt), cover_padded(n, bkt)
    if rg != pd:
        return False
    executed_ragged = n - sum(rg[:-1])          # ragged runs the tail for real
    return executed_ragged == rg[-1]            # padded runs it at bucket width


# --- server plumbing --------------------------------------------------------
def post(port, body, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def model_id(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def n_tok(port, model, text):
    with post(port, {"model": model, "messages": [{"role": "user", "content": text}],
                     "max_tokens": 1, "temperature": 0}) as r:
        return json.load(r)["usage"]["prompt_tokens"]


def answer(port, model, text, max_tokens):
    with post(port, {"model": model, "messages": [{"role": "user", "content": text}],
                     "max_tokens": max_tokens, "temperature": 0}) as r:
        d = json.load(r)
    ch = d["choices"][0]
    return ch["message"]["content"], ch.get("finish_reason", ""), d["usage"]


# --- prompt construction ----------------------------------------------------
def compose(item, fills):
    """head + (filler, needle)* + filler + question + format, in that order."""
    parts = [item["head"]]
    for i, (_, txt) in enumerate(item["needles"]):
        parts.append(FILLER * fills[i])
        parts.append(txt)
    parts.append(FILLER * fills[-1])
    parts.append("\n\n" + item["q"] + item["fmt"])
    return "".join(parts)


def split_fills(total, depths):
    """`total` filler tokens split so needle i sits at depth[i] of the filler."""
    if not depths:
        return [max(total, 0)]
    cuts = [min(max(int(round(d * total)), 0), total) for d in depths]
    cuts = sorted(cuts)
    fills, prev = [], 0
    for c in cuts:
        fills.append(c - prev)
        prev = c
    fills.append(total - prev)
    return fills


def build_exact(port, model, item, target, cache):
    """`item` padded to EXACTLY `target` prompt tokens, VERIFIED against the server.

    A cell that silently misses its length is a cell measured at the wrong plan,
    so this raises instead of returning an approximation.
    """
    key = (item["id"], target)
    if key in cache:
        return cache[key]
    depths = [d for d, _ in item["needles"]]
    total = max(target - n_tok(port, model, compose(item, split_fills(0, depths))), 0)
    got = -1
    for _ in range(10):
        text = compose(item, split_fills(total, depths))
        got = n_tok(port, model, text)
        if got == target:
            cache[key] = text
            return text
        total += target - got
        if total < 0:
            break
    raise SystemExit(f"{item['id']}@{target}: cannot hit the token target exactly (last {got})")


def inject(text, spec, item):
    """Corrupt the prompt the way a LOST CHUNK corrupts it: same token count, so
    the chunk PLAN is identical, but the content of a region is gone.

    This is the gate's own failure proof. A gate never shown failing is how
    `PLOW_FUSE_QUANT` shipped broken past a green gate for weeks.

      drop-tail:N  wipe the N tokens immediately before the question
      drop-mid:N   wipe N tokens at the midpoint
      drop-head:N  wipe the N tokens just after the header
    """
    mode, n = spec.split(":")
    n = int(n)
    # The QUESTION must survive: this injects a loss of retrieved CONTEXT, which
    # is what an unexecuted chunk costs. `rindex("\n\n")` finds the separator
    # inside the format block, not this one, and wiping the question instead
    # makes every item fail for the wrong reason.
    q_at = text.rindex("\n\n" + item["q"])
    body = text[:q_at]
    head_end = body.index("\n\n") + 2
    core = body[head_end:]
    units = re.findall(r"\s*\S+", core)   # whitespace-led units ~ tokens
    if n >= len(units):
        n = max(len(units) - 1, 0)
    if mode == "drop-tail":
        lo = len(units) - n
    elif mode == "drop-mid":
        lo = max((len(units) - n) // 2, 0)
    elif mode == "drop-head":
        lo = 0
    else:
        raise SystemExit(f"unknown --inject mode {mode}")
    units[lo:lo + n] = [FILLER] * n
    return text[:head_end] + "".join(units) + text[q_at:]


# --- grading ----------------------------------------------------------------
ANS_RE = re.compile(r"^[\s>*_#-]*ANSWER\s*[:=]\s*(.+?)\s*$", re.M | re.I)


def norm(s):
    s = s.strip().strip("`*_ ").rstrip(".").strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", "")          # 7,511 == 7511
    return s.lower()


def grade(item, text):
    """-> (correct, formatted, answer_char, detail). `formatted` separates an arm
    that answered WRONG from one that never emitted a checkable answer at all."""
    if item["kind"] == "exact":
        hits = list(ANS_RE.finditer(text))
        if not hits:
            return False, False, -1, "no ANSWER line"
        m = hits[-1]
        got = norm(m.group(1))
        want = norm(item["expect"])
        # A units-carrying answer ("195 pounds") is accepted; a wrong value is not.
        ok = got == want or got.startswith(want) and not got[len(want):len(want) + 1].isdigit()
        return ok, True, m.start(), m.group(1).strip()
    # fields: EVERY key must be present AND right. An omission is a failure.
    miss, last = [], -1
    for k, v in item["expect"]:
        m = None
        for m in re.finditer(rf"{k}\s*[:=]\s*([^\s,;]+)", text, re.I):
            pass
        if m is None:
            miss.append(f"{k}:absent")
        else:
            last = max(last, m.start())
            if norm(m.group(1)) != norm(v):
                miss.append(f"{k}={m.group(1).strip()}!={v}")
    return (not miss), last >= 0, last, ("ok" if not miss else ";".join(miss))


# --- run --------------------------------------------------------------------
def cmd_run(a):
    model = model_id(a.port)
    lens = [int(x) for x in a.lens.split(",")]
    ids = [i for i in a.items.split(",") if i] or list(BY_ID)
    rec = {"arm": a.arm, "model": model, "lens": lens, "inject": a.inject,
           "max_tokens": a.max_tokens, "cells": []}
    cache, t0 = {}, time.perf_counter()
    for iid in ids:
        item = BY_ID[iid]
        for n in lens:
            text = build_exact(a.port, model, item, n, cache)
            if a.inject:
                text = inject(text, a.inject, item)
            txt, fin, usage = answer(a.port, model, text, a.max_tokens)
            ok, formatted, pos, detail = grade(item, txt)
            rec["cells"].append({
                "item": iid, "cls": item["cls"], "tokens": n,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "finish_reason": fin, "correct": ok, "formatted": formatted,
                "answer_char": pos, "answer_frac": (pos / len(txt)) if pos >= 0 and txt else -1.0,
                "expect": item["expect"], "got": detail, "text": txt})
            print(f"  {a.arm} {iid}@{n}: {'OK ' if ok else 'BAD'} "
                  f"{detail[:48]!r} frac={rec['cells'][-1]['answer_frac']:.2f}", flush=True)
    rec["elapsed_s"] = time.perf_counter() - t0
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    n_ok = sum(c["correct"] for c in rec["cells"])
    print(f"wrote {a.out}: {n_ok}/{len(rec['cells'])} correct in {rec['elapsed_s']:.0f}s")


# --- verdict ----------------------------------------------------------------
def binom_ge(k, n):
    """One-sided exact binomial P(X >= k) at p = 0.5 — McNemar without scipy."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)


def cmd_verdict(a):
    base = json.load(open(a.baseline))
    cand = json.load(open(a.candidate))
    bmap = {(c["item"], c["tokens"]): c for c in base["cells"]}
    cmap = {(c["item"], c["tokens"]): c for c in cand["cells"]}
    keys = sorted(set(bmap) & set(cmap))
    errs = []
    if not keys:
        errs.append("no comparable cells")
    if set(bmap) != set(cmap):
        errs.append(f"cell sets differ: base-only {len(set(bmap)-set(cmap))}, "
                    f"cand-only {len(set(cmap)-set(bmap))}")

    b_ok = sum(bmap[k]["correct"] for k in keys)
    c_ok = sum(cmap[k]["correct"] for k in keys)
    reg = [k for k in keys if bmap[k]["correct"] and not cmap[k]["correct"]]
    rep = [k for k in keys if cmap[k]["correct"] and not bmap[k]["correct"]]
    p = binom_ge(len(reg), len(reg) + len(rep))

    print(f"baseline  {base['arm']:<10} {b_ok}/{len(keys)} = {b_ok/len(keys):.1%}")
    print(f"candidate {cand['arm']:<10} {c_ok}/{len(keys)} = {c_ok/len(keys):.1%}"
          + (f"   [inject {cand['inject']}]" if cand.get("inject") else ""))
    print(f"paired: {len(reg)} regressions, {len(rep)} repairs, "
          f"McNemar one-sided p = {p:.4f}")

    print("\nper class            base    cand   reg  rep")
    cls_fail = []
    cls = defaultdict(lambda: [0, 0, 0, 0, 0])
    for k in keys:
        c = cls[bmap[k]["cls"]]
        c[0] += 1
        c[1] += bmap[k]["correct"]
        c[2] += cmap[k]["correct"]
        c[3] += k in reg
        c[4] += k in rep
    for name, (n, bo, co, r, rp) in sorted(cls.items()):
        print(f"  {name:<18} {bo:>2}/{n:<3} {co:>2}/{n:<3} {r:>4} {rp:>4}")
        # A targeted failure must not be diluted away by a large battery: a class
        # the baseline can do and the candidate cannot is a FAIL on its own.
        if bo / n >= 0.8 and co / n < 0.5:
            cls_fail.append(f"class {name}: {bo}/{n} -> {co}/{n}")

    # --- gate validity. A powerless gate must ERROR, never PASS. -------------
    b_fmt = sum(bmap[k]["formatted"] for k in keys) / len(keys) if keys else 0
    b_frac = [bmap[k]["answer_frac"] for k in keys if bmap[k]["answer_frac"] >= 0]
    med_frac = sorted(b_frac)[len(b_frac) // 2] if b_frac else -1
    print(f"\nvalidity: baseline format compliance {b_fmt:.1%}, "
          f"median answer position {med_frac:.0%} into the answer, "
          f"{len(keys)} cells at lens {cand['lens']}")
    if keys and b_ok / len(keys) < a.min_baseline:
        errs.append(f"baseline accuracy {b_ok/len(keys):.1%} < {a.min_baseline:.0%}: "
                    "the battery cannot detect a regression it is already failing")
    if b_fmt < 0.9:
        errs.append(f"baseline format compliance {b_fmt:.1%} < 90%")
    # The whole reason this gate exists: divergence lands ~11% in, so an answer
    # sitting before that is on the identical side of it and carries no signal.
    if med_frac >= 0 and med_frac < 0.25:
        errs.append(f"median answer position {med_frac:.0%} is not LATE enough "
                    "(chunk-plan divergence lands ~11% in)")
    # The ladder is a PROPERTY OF THE BLOB, so a non-GLM battery must pass its
    # own: judging a Gemma-4 cell against GLM's rungs would call a live cell dead.
    bkt = tuple(int(x) for x in a.ladder.split(",")) if a.ladder else LADDER
    on_rung = sorted({c["tokens"] for c in cand["cells"] if carries_no_signal(c["tokens"], bkt)})
    if on_rung:
        errs.append(f"candidate contains ON-RUNG lengths {on_rung}: both arms run the "
                    "same plan AND the same last-chunk row count, so they cannot differ")
    if errs:
        print("\nGATE INVALID:")
        for e in errs:
            print(f"  ! {e}")
        return 2

    fails = []
    if len(reg) > len(rep) and p <= a.alpha:
        fails.append(f"paired regression: {len(reg)} vs {len(rep)}, p={p:.4f} <= {a.alpha}")
    # McNemar alone needs 5 one-sided discordants to reach p <= 0.05, which is
    # lax for a DETERMINISTIC engine where a differing cell is reproducible and
    # not sampling noise. The NET cap is the strict trip; it is net rather than
    # gross so that reassociation flipping cells symmetrically both ways -- which
    # is rewording, not degradation -- does not fire it.
    if len(reg) - len(rep) > a.max_regressions:
        fails.append(f"net regressions {len(reg)-len(rep)} > {a.max_regressions}")
    fails += cls_fail
    if fails:
        print("\nFAIL:")
        for f in fails:
            print(f"  x {f}")
        print("\nregressed cells (baseline correct, candidate wrong):")
        for k in reg:
            print(f"  {k[0]}@{k[1]}: want {bmap[k]['expect']!r} got {cmap[k]['got']!r:.60}")
        return 1
    print("\nPASS: no significant degradation")
    return 0


def cmd_selftest(_):
    """Ground truth recomputed independently of the table above."""
    import statistics
    m = 6 * 60 + 40 + 2 * 60 + 35 + 40 + 3 * 60 + 55
    r = [12, 19, 7, 24, 13]
    want = {
        "hop_train": f"{m//60:02d}:{m%60:02d}",
        "hop_tank": str(int((3 * 1000 * 40 // 100 + 480) * 85 // 100)),
        "hop_money": str(int(240 * 125 // 100 * 80 // 100 - 45)),
        "hop_mod": str((7 ** 4 - 401) % 37),
        "hop_avg": str(int(sum(r) / len(r) * 8 - statistics.median(r))),
        "ndl_sum": str(4817 + 2694),
    }
    bad = [k for k, v in want.items() if BY_ID[k]["expect"] != v]
    n = 1848
    if BY_ID["str_num"]["expect"] != [["DOUBLE", str(2 * n)], ["HALF", str(n // 2)],
                                      ["DIGITSUM", str(sum(map(int, str(n))))],
                                      ["REVERSED", str(n)[::-1]]]:
        bad.append("str_num")
    if BY_ID["str_sq"]["expect"] != [[f"SQ{x}", str(x * x)] for x in range(12, 18)]:
        bad.append("str_sq")
    print("selftest:", "FAIL " + ",".join(bad) if bad else f"OK, {len(ITEMS)} items")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--lens", default=DEFAULT_LENS)
    r.add_argument("--items", default="")
    r.add_argument("--max-tokens", type=int, default=288)
    r.add_argument("--inject", default="", help="drop-tail:N | drop-mid:N | drop-head:N")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("verdict")
    v.add_argument("--baseline", required=True)
    v.add_argument("--candidate", required=True)
    v.add_argument("--alpha", type=float, default=0.05)
    v.add_argument("--min-baseline", type=float, default=0.85)
    v.add_argument("--ladder", default="",
                   help="the blob's prefill bucket ladder, comma-separated "
                        f"(default the GLM-5.2 one: {','.join(map(str, LADDER))})")
    v.add_argument("--max-regressions", type=int, default=2,
                   help="hard cap on NET regressed cells, independent of the p-value")
    v.set_defaults(fn=cmd_verdict)
    s = sub.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
