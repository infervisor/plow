#!/usr/bin/env python3
"""Report ideal tensor traffic and achieved rates from ASR GPU profiles."""
import argparse
import json
from pathlib import Path
import re


def metrics(name, flops, tensor_bytes, gpu_us, bandwidth, compute):
    if gpu_us <= 0 or tensor_bytes <= 0:
        raise ValueError("invalid profile dimensions or duration")
    row = {"name": name, "nominal_flops": flops, "ideal_tensor_bytes": tensor_bytes,
           "ideal_flops_per_byte": flops/tensor_bytes, "gpu_us": gpu_us,
           "achieved_gflops": flops/gpu_us/1000,
           "ideal_bytes_per_second_gb": tensor_bytes/gpu_us/1000}
    if bandwidth:
        row["bandwidth_floor_us"] = tensor_bytes/(bandwidth*1000)
    if compute:
        row["compute_floor_us"] = flops/(compute*1e6)
    if bandwidth and compute:
        row["roofline_floor_us"] = max(row["bandwidth_floor_us"], row["compute_floor_us"])
        row["measured_over_ideal_floor"] = gpu_us/row["roofline_floor_us"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--kind", choices=["encoder", "decoder"], required=True)
    parser.add_argument("--bandwidth-gbps", type=float, help="Explicit sustained calibration, not a chip-name lookup")
    parser.add_argument("--compute-tflops", type=float, help="Explicit compute ceiling; record its provenance")
    parser.add_argument("--ceiling-source", default="none supplied")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for value in [args.bandwidth_gbps, args.compute_tflops]:
        if value is not None and (not 0 < value < float("inf")):
            parser.error("ceilings must be finite and positive")
    rows = []
    if args.kind == "encoder":
        # A run ends at the encoder timing line; use the final complete repetition.
        runs, current = [], []
        for line in args.profile.read_text().splitlines():
            match = re.search(r"asr_kernel=(\w+) params=(\[.*?\]) gpu_us=([0-9.]+)", line)
            if match:
                current.append((match[1], json.loads(match[2]), float(match[3])))
            if line.startswith("encoder ") and current:
                runs.append(current)
                current = []
        if not runs or current:
            raise ValueError("no complete encoder profile or unfinished repetition")
        for name, p, us in runs[-1]:
            if name in ["asr_conv", "asr_conv_tiled"]:
                ci, co, f, t, batch = p
                spatial = ((f+1)//2)*((t+1)//2)
                flops = 2*batch*spatial*co*ci*9
                size = 4*(batch*ci*f*t+co*ci*9+batch*co*spatial+co)
            elif name in ["asr_linear", "asr_linear_tiled"]:
                m, n, k, bias, _ = p
                flops, size = 2*m*n*k, 4*(m*k+n*k+m*n+bias*n)
            else:
                continue
            rows.append(metrics(f"{name} {p}", flops, size, us, args.bandwidth_gbps, args.compute_tflops))
    else:
        runs = json.loads(args.profile.read_text())
        if not runs or not all(r["logits_match"] for r in runs):
            raise ValueError("profile must verify replay logits")
        for inst in runs[-1]["instructions"]:
            name, p = inst["op"], inst["i"]
            m, n, k = p[:3]
            outputs = n
            if name == "GemvQkv":
                n += p[3]+p[4]
                outputs = n
            elif name == "GemvGlu":
                n *= 2
            elif name != "Gemv":
                continue
            rows.append(metrics(f"{inst['index']} {name} {p}", 2*m*n*k,
                2*(m*k+n*k+m*outputs), inst["gpu_us"], args.bandwidth_gbps, args.compute_tflops))
    if not rows:
        raise ValueError("no supported matrix operations")
    args.out.write_text(json.dumps({"profile": str(args.profile), "ceiling_source": args.ceiling_source,
        "bandwidth_gbps": args.bandwidth_gbps, "compute_tflops": args.compute_tflops,
        "limitations": ["Ideal tensor traffic is not measured DRAM traffic; caches/reloads/indexing are not modeled",
                        "Times isolate dispatches and change scheduling; only matrix operations are modeled",
                        "A whole-device roof does not establish single-core utilization"], "rows": rows}, indent=2)+"\n")
    print(f"wrote {len(rows)} matrix observations to {args.out}")


if __name__ == "__main__":
    main()
