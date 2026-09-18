#!/usr/bin/env python3
"""Roofline performance modeling for transformer inference across hardware targets.

Calculates theoretical ceilings (memory bandwidth vs matrix compute), arithmetic
intensity, ridge points, and percent of roofline achieved for prefill (TTFT) and decode (TPOT).

Supported hardware targets:
  - NVIDIA H100 SXM5 (sm_90a)
  - NVIDIA H200 SXM (sm_90a)
  - NVIDIA RTX 5090 (sm_120)
  - NVIDIA RTX 4090 (sm_89)
  - AMD MI300X (gfx942)
  - AMD MI325X (gfx942)
  - AMD MI350X (gfx950)

Usage:
  python3 scripts/campaign/roofline.py --recipe scripts/campaign/recipes/gemma4-12b.h100.bf16-plain.toml
  python3 scripts/campaign/roofline.py --recipe <recipe.toml> --results <results.csv>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    arch: str
    vendor: str
    sm_or_cu_count: int
    clock_boost_mhz: float
    bandwidth_datasheet_gbps: float
    bandwidth_measured_gbps: Optional[float]
    bf16_tflops_dense: float
    fp8_tflops_dense: float
    fp16_tflops_dense: float

    @property
    def bandwidth_for_bound_gbps(self) -> float:
        """Denominator for absolute bounds: measured where available, datasheet otherwise."""
        return self.bandwidth_measured_gbps if self.bandwidth_measured_gbps is not None else self.bandwidth_datasheet_gbps

    def peak_tflops(self, precision: str) -> float:
        p = precision.lower()
        if "fp8" in p:
            return self.fp8_tflops_dense
        if "fp16" in p:
            return self.fp16_tflops_dense
        return self.bf16_tflops_dense

    def ridge_point(self, precision: str) -> float:
        """Arithmetic intensity ridge point (FLOP/byte) where compute bound meets memory bound."""
        compute = self.peak_tflops(precision) * 1e12
        bw = self.bandwidth_for_bound_gbps * 1e9
        return compute / bw if bw > 0 else 0.0


HARDWARE_REGISTRY: Dict[str, HardwareSpec] = {
    "h100 sxm5": HardwareSpec(
        name="H100 SXM5",
        arch="sm_90a",
        vendor="nvidia",
        sm_or_cu_count=132,
        clock_boost_mhz=1980.0,
        bandwidth_datasheet_gbps=3352.0,
        bandwidth_measured_gbps=None,
        bf16_tflops_dense=989.0,
        fp8_tflops_dense=1979.0,
        fp16_tflops_dense=989.0,
    ),
    "h200 sxm": HardwareSpec(
        name="H200 SXM",
        arch="sm_90a",
        vendor="nvidia",
        sm_or_cu_count=132,
        clock_boost_mhz=1980.0,
        bandwidth_datasheet_gbps=4800.0,
        bandwidth_measured_gbps=None,
        bf16_tflops_dense=989.0,
        fp8_tflops_dense=1979.0,
        fp16_tflops_dense=989.0,
    ),
    "h100 pcie": HardwareSpec(
        name="H100 PCIe",
        arch="sm_90a",
        vendor="nvidia",
        sm_or_cu_count=114,
        clock_boost_mhz=1755.0,
        bandwidth_datasheet_gbps=2039.0,
        bandwidth_measured_gbps=None,
        bf16_tflops_dense=756.0,
        fp8_tflops_dense=1513.0,
        fp16_tflops_dense=756.0,
    ),
    "rtx 5090": HardwareSpec(
        name="RTX 5090",
        arch="sm_120",
        vendor="nvidia",
        sm_or_cu_count=170,
        clock_boost_mhz=2407.0,
        bandwidth_datasheet_gbps=1792.0,
        bandwidth_measured_gbps=None,
        bf16_tflops_dense=835.0,
        fp8_tflops_dense=1670.0,
        fp16_tflops_dense=835.0,
    ),
    "rtx 4090": HardwareSpec(
        name="RTX 4090",
        arch="sm_89",
        vendor="nvidia",
        sm_or_cu_count=128,
        clock_boost_mhz=2520.0,
        bandwidth_datasheet_gbps=1008.0,
        bandwidth_measured_gbps=None,
        bf16_tflops_dense=330.0,
        fp8_tflops_dense=660.0,
        fp16_tflops_dense=330.0,
    ),
    "mi300x": HardwareSpec(
        name="MI300X",
        arch="gfx942",
        vendor="amd",
        sm_or_cu_count=304,
        clock_boost_mhz=2100.0,
        bandwidth_datasheet_gbps=5325.0,
        bandwidth_measured_gbps=4091.9,  # Measured sustainable read bandwidth
        bf16_tflops_dense=1307.0,
        fp8_tflops_dense=2614.0,
        fp16_tflops_dense=1307.0,
    ),
    "mi325x": HardwareSpec(
        name="MI325X",
        arch="gfx942",
        vendor="amd",
        sm_or_cu_count=304,
        clock_boost_mhz=2100.0,
        bandwidth_datasheet_gbps=6000.0,
        bandwidth_measured_gbps=4164.0,
        bf16_tflops_dense=1307.0,
        fp8_tflops_dense=2614.0,
        fp16_tflops_dense=1307.0,
    ),
    "mi350x": HardwareSpec(
        name="MI350X",
        arch="gfx950",
        vendor="amd",
        sm_or_cu_count=304,
        clock_boost_mhz=2400.0,
        bandwidth_datasheet_gbps=8000.0,
        bandwidth_measured_gbps=6200.0,
        bf16_tflops_dense=2300.0,
        fp8_tflops_dense=4600.0,
        fp16_tflops_dense=2300.0,
    ),
}


def lookup_gpu(name_or_arch: str) -> HardwareSpec:
    q = name_or_arch.strip().lower()
    for k, spec in HARDWARE_REGISTRY.items():
        if k in q or spec.name.lower() in q or spec.arch.lower() == q:
            return spec
    # Fallback fuzzy match
    if "h100" in q:
        return HARDWARE_REGISTRY["h100 sxm5"]
    if "h200" in q:
        return HARDWARE_REGISTRY["h200 sxm"]
    if "5090" in q:
        return HARDWARE_REGISTRY["rtx 5090"]
    if "4090" in q:
        return HARDWARE_REGISTRY["rtx 4090"]
    if "mi300" in q or "gfx942" in q:
        return HARDWARE_REGISTRY["mi300x"]
    if "mi350" in q or "gfx950" in q:
        return HARDWARE_REGISTRY["mi350x"]
    # Generic default: H100 SXM5
    return HARDWARE_REGISTRY["h100 sxm5"]


@dataclass
class ModelSpec:
    name: str
    active_params: int
    layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    precision: str = "bf16"

    @property
    def elem_bytes(self) -> float:
        p = self.precision.lower()
        if "fp8" in p or "int8" in p:
            return 1.0
        if "fp4" in p:
            return 0.5
        return 2.0  # bf16 / fp16

    @property
    def weight_bytes(self) -> int:
        return int(self.active_params * self.elem_bytes)

    @property
    def kv_bytes_per_token(self) -> int:
        return int(2 * self.layers * self.num_kv_heads * self.head_dim * self.elem_bytes)


KNOWN_MODELS: Dict[str, ModelSpec] = {
    "gemma4-12b": ModelSpec(
        name="Gemma-4-12B",
        active_params=12_000_000_000,
        layers=40,
        hidden_size=3840,
        num_heads=30,
        num_kv_heads=10,
        head_dim=256,
    ),
    "gemma4-26b": ModelSpec(
        name="Gemma-4-26B",
        active_params=26_000_000_000,
        layers=46,
        hidden_size=5120,
        num_heads=40,
        num_kv_heads=16,
        head_dim=256,
    ),
    "glm-5.3": ModelSpec(
        name="GLM-5.3",
        active_params=14_000_000_000,  # ~14B active parameters per token (MLA+MoE)
        layers=40,
        hidden_size=6144,
        num_heads=64,
        num_kv_heads=64,
        head_dim=256,
    ),
    "llama3-70b": ModelSpec(
        name="Llama-3-70B",
        active_params=70_000_000_000,
        layers=80,
        hidden_size=8192,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
    ),
}


def lookup_model(recipe_data: dict) -> ModelSpec:
    cell = recipe_data.get("cell", {})
    name = cell.get("name", "")
    model_str = cell.get("model", "")
    precision = cell.get("precision", "bf16")

    # Match by known model substrings
    for key, spec in KNOWN_MODELS.items():
        if key in name.lower() or key in model_str.lower():
            return ModelSpec(
                name=spec.name,
                active_params=spec.active_params,
                layers=spec.layers,
                hidden_size=spec.hidden_size,
                num_heads=spec.num_heads,
                num_kv_heads=spec.num_kv_heads,
                head_dim=spec.head_dim,
                precision=precision,
            )

    # Estimate based on name (e.g. 12b -> 12B)
    import re
    m = re.search(r"(\d+)b", (name + model_str).lower())
    param_billions = float(m.group(1)) if m else 12.0
    active_params = int(param_billions * 1e9)
    return ModelSpec(
        name=name or f"Model-{int(param_billions)}B",
        active_params=active_params,
        layers=40,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        precision=precision,
    )


@dataclass
class DecodeAnalysis:
    batch_size: int
    ctx_len: int
    weight_bytes: int
    kv_bytes: int
    total_bytes: int
    flops: int
    arithmetic_intensity: float
    roofline_ms: float
    memory_roof_ms: float
    compute_roof_ms: float
    bound: str
    measured_tpot_ms: Optional[float] = None
    achieved_gbps: Optional[float] = None
    pct_roofline: Optional[float] = None
    headroom: Optional[float] = None


@dataclass
class PrefillAnalysis:
    prompt_tokens: int
    total_flops: int
    total_bytes: int
    arithmetic_intensity: float
    ridge_point: float
    roofline_ms: float
    memory_roof_ms: float
    compute_roof_ms: float
    bound: str
    measured_ttft_ms: Optional[float] = None
    achieved_tflops: Optional[float] = None
    pct_roofline: Optional[float] = None
    headroom: Optional[float] = None


def analyze_decode(
    gpu: HardwareSpec,
    model: ModelSpec,
    batch_size: int,
    ctx_len: int,
    measured_tpot_ms: Optional[float] = None,
) -> DecodeAnalysis:
    weight_bytes = model.weight_bytes
    kv_bytes = model.kv_bytes_per_token * batch_size * ctx_len
    total_bytes = weight_bytes + kv_bytes
    flops = 2 * model.active_params * batch_size
    ai = flops / total_bytes if total_bytes > 0 else 0.0

    bw_gbps = gpu.bandwidth_for_bound_gbps
    peak_tf = gpu.peak_tflops(model.precision)
    mem_roof_ms = (total_bytes / (bw_gbps * 1e9)) * 1e3
    comp_roof_ms = (flops / (peak_tf * 1e12)) * 1e3
    roof_ms = max(mem_roof_ms, comp_roof_ms)
    bound = "Memory" if mem_roof_ms >= comp_roof_ms else "Compute"

    achieved_gbps = None
    pct_roofline = None
    headroom = None
    if measured_tpot_ms is not None and measured_tpot_ms > 0:
        achieved_gbps = (total_bytes / (measured_tpot_ms * 1e-3)) / 1e9
        pct_roofline = (achieved_gbps / bw_gbps) * 100.0 if bw_gbps > 0 else 0.0
        headroom = measured_tpot_ms / roof_ms if roof_ms > 0 else 0.0

    return DecodeAnalysis(
        batch_size=batch_size,
        ctx_len=ctx_len,
        weight_bytes=weight_bytes,
        kv_bytes=kv_bytes,
        total_bytes=total_bytes,
        flops=flops,
        arithmetic_intensity=ai,
        roofline_ms=roof_ms,
        memory_roof_ms=mem_roof_ms,
        compute_roof_ms=comp_roof_ms,
        bound=bound,
        measured_tpot_ms=measured_tpot_ms,
        achieved_gbps=achieved_gbps,
        pct_roofline=pct_roofline,
        headroom=headroom,
    )


def analyze_prefill(
    gpu: HardwareSpec,
    model: ModelSpec,
    prompt_tokens: int,
    measured_ttft_ms: Optional[float] = None,
) -> PrefillAnalysis:
    gemm_flops = 2 * model.active_params * prompt_tokens
    # Causal attention FLOPs: 2 * layers * heads * head_dim * T^2
    attn_flops = 2 * model.layers * model.num_heads * model.head_dim * (prompt_tokens * prompt_tokens)
    total_flops = gemm_flops + attn_flops

    weight_bytes = model.weight_bytes
    act_bytes = int(2.0 * model.layers * model.hidden_size * prompt_tokens * model.elem_bytes)
    total_bytes = weight_bytes + act_bytes

    ai = total_flops / total_bytes if total_bytes > 0 else 0.0
    ridge = gpu.ridge_point(model.precision)

    bw_gbps = gpu.bandwidth_for_bound_gbps
    peak_tf = gpu.peak_tflops(model.precision)
    mem_roof_ms = (total_bytes / (bw_gbps * 1e9)) * 1e3
    comp_roof_ms = (total_flops / (peak_tf * 1e12)) * 1e3
    roof_ms = max(mem_roof_ms, comp_roof_ms)
    bound = "Compute" if ai >= ridge else "Memory"

    achieved_tflops = None
    pct_roofline = None
    headroom = None
    if measured_ttft_ms is not None and measured_ttft_ms > 0:
        achieved_tflops = (total_flops / (measured_ttft_ms * 1e-3)) / 1e12
        pct_roofline = (achieved_tflops / peak_tf) * 100.0 if peak_tf > 0 else 0.0
        headroom = measured_ttft_ms / roof_ms if roof_ms > 0 else 0.0

    return PrefillAnalysis(
        prompt_tokens=prompt_tokens,
        total_flops=total_flops,
        total_bytes=total_bytes,
        arithmetic_intensity=ai,
        ridge_point=ridge,
        roofline_ms=roof_ms,
        memory_roof_ms=mem_roof_ms,
        compute_roof_ms=comp_roof_ms,
        bound=bound,
        measured_ttft_ms=measured_ttft_ms,
        achieved_tflops=achieved_tflops,
        pct_roofline=pct_roofline,
        headroom=headroom,
    )


def diagnose_bottleneck(decode: DecodeAnalysis, prefill: PrefillAnalysis) -> str:
    """Identify the primary hardware/kernel bottleneck based on achieved roofline."""
    diagnostics = []
    if decode.pct_roofline is not None:
        if decode.pct_roofline >= 80.0:
            diagnostics.append(f"Decode ({decode.pct_roofline:.1f}% mem roof): saturated memory subsystem; near optimal")
        elif decode.pct_roofline >= 55.0:
            diagnostics.append(f"Decode ({decode.pct_roofline:.1f}% mem roof): good utilization; bound by GEMV tile shape / row blocking")
        else:
            diagnostics.append(f"Decode ({decode.pct_roofline:.1f}% mem roof): low bandwidth efficiency; check dispatch overhead or wave tail quantization")

    if prefill.pct_roofline is not None:
        if prefill.pct_roofline >= 70.0:
            diagnostics.append(f"Prefill ({prefill.pct_roofline:.1f}% compute roof): high compute saturation; near cuBLASLt ceiling")
        elif prefill.pct_roofline >= 40.0:
            diagnostics.append(f"Prefill ({prefill.pct_roofline:.1f}% compute roof): moderate compute efficiency; check attention/GLU fusion or wave alignment")
        else:
            diagnostics.append(f"Prefill ({prefill.pct_roofline:.1f}% compute roof): compute under-utilized; small chunk size or unoptimized GEMM kernel")

    return " | ".join(diagnostics) if diagnostics else "No measured timings provided."


def generate_roofline_report(
    recipe_path: Path,
    results_path: Optional[Path] = None,
) -> str:
    with open(recipe_path, "rb") as f:
        r = tomllib.load(f)

    cell = r.get("cell", {})
    gpu = lookup_gpu(cell.get("gpu", "") or cell.get("arch", ""))
    model = lookup_model(r)

    measured_rows: Dict[Tuple[int, int], Dict[str, float]] = {}
    if results_path and results_path.is_file():
        with open(results_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    k = (int(row["input_len"]), int(row["concurrency"]))
                    measured_rows[k] = {
                        "ttft_ms": float(row["ttft_ms"]),
                        "tpot_ms": float(row["tpot_ms"]),
                        "out_tok_s": float(row.get("out_tok_s", 0.0)),
                    }
                except (KeyError, ValueError):
                    continue

    out = []
    out.append(f"=== Roofline Analysis: {cell.get('name', recipe_path.stem)} ===")
    out.append(f"Target GPU  : {gpu.name} ({gpu.arch})")
    out.append(f"Memory BW   : {gpu.bandwidth_for_bound_gbps:.1f} GB/s " +
               (f"(measured read; datasheet={gpu.bandwidth_datasheet_gbps:.0f} GB/s)" if gpu.bandwidth_measured_gbps else "(datasheet peak)"))
    out.append(f"Peak Compute: {gpu.peak_tflops(model.precision):.1f} TFLOP/s ({model.precision.upper()} dense)")
    out.append(f"Ridge Point : {gpu.ridge_point(model.precision):.1f} FLOP/byte")
    out.append(f"Model       : {model.name} ({model.active_params/1e9:.1f}B params, {model.weight_bytes/1e9:.2f} GB weights)")
    out.append("")

    # If results exist, evaluate measured rows
    if measured_rows:
        out.append(f"{'in':>5} {'C':>2} | {'TTFT ms':>8} {'%roof_pf':>8} {'TF/s':>7} | {'TPOT ms':>8} {'%roof_dec':>9} {'GB/s':>7} | {'Limiter':<12}")
        out.append("-" * 78)
        for (isl, c), m in sorted(measured_rows.items()):
            dec = analyze_decode(gpu, model, batch_size=c, ctx_len=isl, measured_tpot_ms=m["tpot_ms"])
            pf = analyze_prefill(gpu, model, prompt_tokens=isl, measured_ttft_ms=m["ttft_ms"])
            out.append(
                f"{isl:>5} {c:>2} | {m['ttft_ms']:>8.2f} {pf.pct_roofline:>7.1f}% {pf.achieved_tflops:>7.1f} | "
                f"{m['tpot_ms']:>8.2f} {dec.pct_roofline:>8.1f}% {dec.achieved_gbps:>7.1f} | {pf.bound} / {dec.bound}"
            )
        out.append("")
        out.append("Bottleneck Diagnosis:")
        # Use largest input length and C=1 for diagnosis
        c1_keys = [k for k in measured_rows if k[1] == 1]
        sample_key = max(c1_keys) if c1_keys else list(measured_rows.keys())[0]
        sample_m = measured_rows[sample_key]
        sample_dec = analyze_decode(gpu, model, batch_size=sample_key[1], ctx_len=sample_key[0], measured_tpot_ms=sample_m["tpot_ms"])
        sample_pf = analyze_prefill(gpu, model, prompt_tokens=sample_key[0], measured_ttft_ms=sample_m["ttft_ms"])
        out.append("  " + diagnose_bottleneck(sample_dec, sample_pf))
    else:
        # Theoretical ceilings across typical ladders
        out.append("Theoretical Ceilings (Zero Overhead Roofline Floor):")
        out.append(f"{'in':>5} {'C':>2} | {'TTFT Floor':>10} {'Pf FLOPs':>10} {'AI':>6} | {'TPOT Floor':>10} {'Dec Bytes':>10} | {'Limiter':<8}")
        out.append("-" * 76)
        test_points = [(128, 1), (1024, 1), (4096, 1), (1024, 4), (4096, 4), (4096, 16)]
        for isl, c in test_points:
            dec = analyze_decode(gpu, model, batch_size=c, ctx_len=isl)
            pf = analyze_prefill(gpu, model, prompt_tokens=isl)
            out.append(
                f"{isl:>5} {c:>2} | {pf.roofline_ms:>8.2f} ms {pf.total_flops/1e12:>9.2f}T {pf.arithmetic_intensity:>6.1f} | "
                f"{dec.roofline_ms:>8.2f} ms {dec.total_bytes/1e9:>9.2f}G | {pf.bound}/{dec.bound}"
            )

    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", help="Path to campaign recipe.toml")
    ap.add_argument("--results", help="Path to results.csv")
    ap.add_argument("--gpu", default="H100 SXM5", help="GPU name or arch")
    ap.add_argument("--model", default="gemma4-12b", help="Model slug or name")
    ap.add_argument("--precision", default="bf16", help="Precision (bf16, fp8, fp16)")
    args = ap.parse_args()

    if args.recipe:
        print(generate_roofline_report(Path(args.recipe), Path(args.results) if args.results else None))
    else:
        gpu = lookup_gpu(args.gpu)
        model = KNOWN_MODELS.get(args.model, ModelSpec(
            name=args.model,
            active_params=12_000_000_000,
            layers=40,
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            precision=args.precision,
        ))
        print(f"=== Theoretical Roofline: {model.name} on {gpu.name} ({model.precision.upper()}) ===")
        print(f"Memory Bandwidth: {gpu.bandwidth_for_bound_gbps:.1f} GB/s")
        print(f"Compute Peak    : {gpu.peak_tflops(model.precision):.1f} TFLOP/s")
        print(f"Ridge Point     : {gpu.ridge_point(model.precision):.1f} FLOP/byte\n")
        print(f"{'in':>5} {'C':>2} | {'TTFT Floor':>10} {'Pf AI':>8} {'Limiter':<8} | {'TPOT Floor':>10} {'Limiter':<8}")
        print("-" * 65)
        for isl, c in [(128, 1), (1024, 1), (4096, 1), (1024, 4), (4096, 4), (4096, 16)]:
            dec = analyze_decode(gpu, model, batch_size=c, ctx_len=isl)
            pf = analyze_prefill(gpu, model, prompt_tokens=isl)
            print(f"{isl:>5} {c:>2} | {pf.roofline_ms:>8.2f} ms {pf.arithmetic_intensity:>8.1f} {pf.bound:<8} | {dec.roofline_ms:>8.2f} ms {dec.bound:<8}")


if __name__ == "__main__":
    main()
