#!/usr/bin/env bash
# thinking_gates.sh <assets> <objdir> <out> — GPU soundness gates for thinking control
# (`serve::think`, docs/serving-openai-compat.md "Thinking control").
#
#   F1  temperature 0, a budget, a forced end and newline grace 4: the output (content, reasoning,
#       x_plow_thinking) is byte-identical across the decode paths that chain tokens on the device
#       (pipelined lookahead, speculative verify, multistep quantum) and the plain one; and the same
#       prompts WITHOUT a budget differ (else the budget was ignored).
#   F2  presence_penalty 0.5 + repetition_penalty 1.1 under forcing: whether the paths differ is
#       recorded, not failed (the device penalty history is reseeded after a forced id).
#   F4  `seed` set, budget set, temperature 1: two runs are identical.
#   F3  only when PLOWRT_BASE_BIN names a build without thinking control: no-budget TPOT at
#       c1/c8/c32, this binary vs that one, via pb_bench (judge it against the repeat spread).
#
# Every arm is its own server process. The base serve environment (the production recipe's
# `serve_env`) goes on the queued command; arms override only PLOW_DECODE_PIPELINE, PLOW_SPEC and
# PLOW_MULTISTEP. Run it through the queue:
#   scripts/bench/gpuq.py submit thinking-gates 1 env <recipe serve_env> PLOWRT_BIN=<plowrt> \
#       scripts/bench/thinking_gates.sh <assets> <objdir> <scratch out>
# Exit 0 = F1 and F4 pass; 1 = a gate failed; 2 = refused (environment).
set -euo pipefail
# shellcheck source=plowbench.sh
source "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/scripts/bench/plowbench.sh"

[ $# -eq 3 ] || { sed -n '2,23p' "$0"; exit 2; }
assets=$1 objdir=$2 out=$3
rt=${PLOWRT_BIN:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/target/release/plowrt}
pb_require_nix || exit 2
pb_check_plowrt "$rt" || exit 2
mkdir -p "$out"
trap pb_serve_stop EXIT

BUDGET=${BUDGET:-32}
END=${REASONING_END:-$'\nStop. Replying now, in the caller\'s language, one question only.<channel|>'}

# name|env overrides, base first. THINKING_GATE_ARMS replaces them, `;`-separated (a packet
# without a drafter has no speculative arm).
ARMS=(
    "base|PLOW_DECODE_PIPELINE=1 PLOW_SPEC=1 PLOW_MULTISTEP=0"
    "nopipe|PLOW_DECODE_PIPELINE=0 PLOW_SPEC=1 PLOW_MULTISTEP=0"
    "nospec|PLOW_DECODE_PIPELINE=1 PLOW_SPEC=0 PLOW_MULTISTEP=0"
    "multistep|PLOW_DECODE_PIPELINE=0 PLOW_SPEC=0 PLOW_MULTISTEP=8"
    "plain|PLOW_DECODE_PIPELINE=0 PLOW_SPEC=0 PLOW_MULTISTEP=0"
)
if [ -n "${THINKING_GATE_ARMS:-}" ]; then
    IFS=';' read -r -a ARMS <<< "$THINKING_GATE_ARMS"
fi

serve_arm() { # <name> [VAR=value ...] — the arm's environment reaches only its server
    local name=$1 kv port
    shift
    for kv in "$@"; do local -x "$kv"; done
    port=$(pb_free_port)
    pb_serve_start "$rt" "$assets" "$objdir" "$port" "$out/$name.serve.log"
    pb_serve_wait 1800
}

requests() { # <name> <model> <first: 1|0> — every gate's requests against the running arm
    python3 - "$PB_SERVER_PORT" "$2" "$BUDGET" "$END" "$out/$1.json" "$3" <<'PYEOF'
import json, sys, urllib.request
port, model, budget, end, dst, first = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6] == "1"
prompts = [
    "Hello, who is this?",
    "मेरा पार्सल कब आएगा?",
    "আমি রাহুল বলছি, আপনি কে?",
    "Explain in one line why the sky is blue.",
]
def ask(body):
    body = {"model": model, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": True}, **body}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"content-type": "application/json"})
    v = json.load(urllib.request.urlopen(req, timeout=600))
    c = v["choices"][0]
    return {"content": c["message"].get("content"), "reasoning": c["message"].get("reasoning"),
            "finish": c["finish_reason"], "thinking": c.get("x_plow_thinking")}
res = {}
for i, p in enumerate(prompts):
    m = [{"role": "user", "content": p}]
    budget_body = {"messages": m, "temperature": 0, "thinking_token_budget": budget, "reasoning_end": end}
    res[f"f1.{i}"] = ask(budget_body)
    res[f"f2.{i}"] = ask({**budget_body, "presence_penalty": 0.5, "repetition_penalty": 1.1})
    if first:
        res[f"f1.nobudget.{i}"] = ask({"messages": m, "temperature": 0})
        seeded = {"messages": m, "temperature": 1.0, "seed": 7, "thinking_token_budget": budget, "reasoning_end": end}
        res[f"f4.{i}.a"] = ask(seeded)
        res[f"f4.{i}.b"] = ask(seeded)
json.dump(res, open(dst, "w"), ensure_ascii=False, indent=1)
PYEOF
}

# The served name, for the defaults file that sets the newline grace.
# shellcheck disable=SC2086
serve_arm probe ${ARMS[0]#*|} || exit 1
model=$(pb_model_id)
pb_serve_stop
printf '{"%s": {"newline_grace": 4}}\n' "$model" > "$out/chat-defaults.json"

first=1
for arm in "${ARMS[@]}"; do
    name=${arm%%|*}
    # shellcheck disable=SC2086
    serve_arm "$name" ${arm#*|} "PLOW_CHAT_DEFAULTS=$out/chat-defaults.json" || exit 1
    requests "$name" "$model" "$first"
    first=0
    pb_serve_stop
done

if [ -n "${PLOWRT_BASE_BIN:-}" ]; then
    for tag in new base; do
        bin=$rt
        [ "$tag" = base ] && bin=$PLOWRT_BASE_BIN
        port=$(pb_free_port)
        pb_serve_start "$bin" "$assets" "$objdir" "$port" "$out/f3.$tag.serve.log"
        pb_serve_wait 1800 || exit 1
        for c in 1 8 32; do
            for rep in 1 2; do
                pb_bench "$out/f3" "$tag.c$c.r$rep" "$model" "$c" $((c * 8)) 1024 256 --temperature 0
            done
        done
        pb_serve_stop
    done
fi

python3 - "$out" "${ARMS[@]%%|*}" <<'PYEOF'
import json, sys
out, arms = sys.argv[1], sys.argv[2:]
r = {a: json.load(open(f"{out}/{a}.json")) for a in arms}
base = r[arms[0]]
fail, lines = 0, []
def same(key):
    return [a for a in arms[1:] if r[a][key] != base[key]]
for key in sorted(k for k in base if k.startswith("f1.") and "nobudget" not in k):
    bad = same(key)
    forced = base[key]["thinking"] and base[key]["thinking"].get("forced")
    lines.append(f"F1 {key}: {'PASS' if not bad else 'FAIL differs on ' + ','.join(bad)} (forced={forced})")
    fail += bool(bad)
diff = [k for k in base if k.startswith("f1.nobudget.") and base[k] != base["f1." + k.split(".")[-1]]]
lines.append(f"F1 control (no budget differs): {'PASS' if diff else 'FAIL: the budget changed nothing'}")
fail += not diff
for key in sorted(k for k in base if k.startswith("f2.")):
    lines.append(f"F2 {key}: paths differing = {same(key) or 'none'} (recorded)")
for key in sorted({k.rsplit('.', 1)[0] for k in base if k.startswith("f4.")}):
    ok = base[key + ".a"] == base[key + ".b"]
    lines.append(f"F4 {key}: {'PASS' if ok else 'FAIL: seeded runs differ'}")
    fail += not ok
open(f"{out}/gates.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if fail else 0)
PYEOF
