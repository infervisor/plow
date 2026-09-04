#!/usr/bin/env python3
import argparse
import math


def max_route_distribution(experts, topk, ep):
    sizes = [len(range(g, experts, ep)) for g in range(ep)]
    states = {(0, 0): 1}
    for size in sizes:
        nxt = {}
        for (used, old_max), ways in states.items():
            for count in range(min(size, topk - used) + 1):
                key = used + count, max(old_max, count)
                nxt[key] = nxt.get(key, 0) + ways * math.comb(size, count)
        states = nxt
    total = math.comb(experts, topk)
    dist = {maximum: ways / total for (used, maximum), ways in states.items() if used == topk}
    assert abs(sum(dist.values()) - 1.0) < 1e-12
    return dist


def quantile(dist, q):
    cumulative = 0.0
    for value, probability in sorted(dist.items()):
        cumulative += probability
        if cumulative >= q:
            return value
    raise AssertionError("incomplete distribution")


def resident_bytes(experts, hidden, intermediate, world, ep, layers, stage2):
    tp = world // ep
    owned = experts // ep
    local_i = intermediate // tp
    payload = hidden * local_i // 2
    scales = hidden * (local_i // 32)
    primary = layers * owned * 3 * (payload + scales)
    shuffled = 0
    if stage2:
        padded_scales = math.ceil(hidden / 256) * 256 * math.ceil(local_i / 256) * 256 // 32
        shuffled = layers * owned * (payload + padded_scales)
    return primary, shuffled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=896)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--hidden", type=int, default=3584)
    p.add_argument("--intermediate", type=int, default=3072)
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--layers", type=int, default=92)
    p.add_argument("--stage2-view", action="store_true")
    args = p.parse_args()
    for ep in (d for d in range(1, args.world + 1) if args.world % d == 0):
        if args.experts % ep or args.intermediate % (args.world // ep):
            continue
        dist = max_route_distribution(args.experts, args.topk, ep)
        mean = sum(value * probability for value, probability in dist.items())
        p50, p95, p99 = (quantile(dist, q) for q in (0.50, 0.95, 0.99))
        primary, shuffled = resident_bytes(
            args.experts, args.hidden, args.intermediate, args.world, ep,
            args.layers, args.stage2_view,
        )
        print(
            f"EP{ep}xTP{args.world // ep} I_local={args.intermediate // (args.world // ep)} "
            f"owned={args.experts // ep} max_routes_mean={mean:.6f} "
            f"p50={p50} p95={p95} p99={p99} "
            f"tail_p50={p50 / (args.topk / ep):.3f}x "
            f"resident_primary={primary} resident_stage2={shuffled} total={primary + shuffled}"
        )


if __name__ == "__main__":
    main()
