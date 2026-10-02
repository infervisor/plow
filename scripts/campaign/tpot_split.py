"""Decompose the TPOT gap: fixed per-step overhead vs per-context-token KV traversal.

TPOT(ctx) ~ fixed + slope*ctx. Fit both stacks on the measured ladder and compare the two terms
separately, because they have DIFFERENT fixes: a fixed offset is per-step work (megakernel entry,
router/MoE launch, norms) and a slope is KV bytes per token / achieved KV bandwidth.

Then convert the slope into achieved KV bandwidth from the checkpoint's own KV geometry, so the
question "how much headroom is left" gets a number rather than a ratio.
"""
import json

CK = "/opt/dlami/nvme/hf-cache/hub/gemma-4-26b-a4b-it"
cfg = json.loads(open(f"{CK}/config.json").read())
t = cfg.get("text_config", cfg)

layers = int(t["num_hidden_layers"])
n_kv = int(t.get("num_key_value_heads", t["num_attention_heads"]))
hd = int(t.get("head_dim", t["hidden_size"] // t["num_attention_heads"]))
print(f"26B-A4B: layers={layers} n_kv_heads={n_kv} head_dim={hd}")
# Gemma-4 alternates sliding (hd256/GQA2) and full (hd512/GQA8) per the recipe header; the config
# gives one nominal geometry, so report the nominal bytes/token and say so.
kv_per_tok = layers * n_kv * hd * 2 * 2      # K and V, bf16
print(f"nominal KV bytes per token (all layers, K+V, bf16): {kv_per_tok/1024:.1f} KiB")
print("  (nominal: Gemma-4 alternates sliding hd256/GQA2 and full hd512/GQA8, and the sliding")
print("   layers cap at the 1024 window, so the EFFECTIVE bytes/token is lower at long ctx.)")
print()

CELLS = {
    1: {1024: (5.51, 5.07), 2048: (5.53, 5.08), 4096: (5.54, 5.09),
        8192: (5.59, 5.09), 15000: (5.66, 5.09)},
    4: {1024: (8.53, 7.56), 2048: (8.85, 7.55), 4096: (9.59, 7.94),
        8192: (12.17, 8.56), 15000: (17.51, 10.83)},
    16: {1024: (13.68, 10.12), 2048: (16.38, 10.72), 4096: (21.98, 13.72),
         8192: (35.13, 20.97), 15000: (55.11, 34.92)},
    32: {1024: (13.60, 13.51), 2048: (16.73, 16.15), 4096: (22.39, 22.03),
         8192: (35.75, 36.33), 15000: (55.07, 62.73)},
}

BW = 3352e9  # H100 SXM5 GB/s


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


print(f"{'C':>3} | {'fixed ms plow/vllm':>22} {'slope us per 1k ctx':>26} "
      f"{'slope ratio':>12} | {'gap@1k':>8} {'gap@15k':>8} {'fixed share of gap@1k':>22}")
for C, cells in CELLS.items():
    xs = sorted(cells)
    p = [cells[x][0] for x in xs]
    v = [cells[x][1] for x in xs]
    fp, sp = fit(xs, p)
    fv, sv = fit(xs, v)
    g1 = cells[1024][0] - cells[1024][1]
    g15 = cells[15000][0] - cells[15000][1]
    share = 100.0 * (fp - fv) / g1 if g1 else float("nan")
    print(f"{C:>3} | {fp:9.2f} /{fv:9.2f} {sp*1000:11.1f} /{sv*1000:11.1f} "
          f"{sp/sv:12.2f} | {g1:8.2f} {g15:8.2f} {share:21.0f}%")

print()
print("Achieved KV bandwidth implied by the slope (B sequences each growing by 1 token of ctx):")
print(f"{'C':>3} | {'plow GB/s':>11} {'vllm GB/s':>11} {'plow % of 3352':>16} {'vllm % of 3352':>16}")
for C, cells in CELLS.items():
    xs = sorted(cells)
    _, sp = fit(xs, [cells[x][0] for x in xs])
    _, sv = fit(xs, [cells[x][1] for x in xs])
    # slope is ms of step time per +1 ctx token, with C sequences live -> C*kv_per_tok bytes added
    def gbs(slope_ms):
        return (C * kv_per_tok) / (slope_ms * 1e-3) / 1e9 if slope_ms > 0 else float("nan")
    print(f"{C:>3} | {gbs(sp):11.0f} {gbs(sv):11.0f} "
          f"{100*gbs(sp)*1e9/BW:15.1f}% {100*gbs(sv)*1e9/BW:15.1f}%")
print()
print("NB C32 on this packet is slot-capped at 16, so its row describes 16 live slots, not 32;")
print("its 'wins' at 8192/15000 are vLLM degrading, not plow improving. Read C1-C16.")
