#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys


SUPPORTED = {
    "PLOW_DOP_GEMV": ("gemv", "None"),
    "PLOW_DOP_GEMV_GLU": ("gemvglu", "None"),
    "PLOW_DOP_GEMV_QKV": ("gemvqkv", "None"),
    "PLOW_DOP_GEMV_QKVG": ("gemvqkvg", "None"),
    "PLOW_DOP_GEMV_MXFP4": ("gemv", "Mxfp4"),
    "PLOW_DOP_GEMV_GLU_MXFP4": ("gemvglu", "Mxfp4"),
}
SYMBOL_FAMILY = {
    "gemv": ("gemv", "None"),
    "gemv_glu": ("gemvglu", "None"),
    "gemv_qkv": ("gemvqkv", "None"),
    "gemv_qkvg": ("gemvqkvg", "None"),
    "gemv_mxfp4": ("gemv", "Mxfp4"),
    "gemv_glu_mxfp4": ("gemvglu", "Mxfp4"),
}
FAMILY_SYMBOL = {value: key for key, value in SYMBOL_FAMILY.items()}
BUCKETS = (1, 2, 4, 8, 16)
DEMAND_MMS = (1, 2, 4, 8, 16, 32, 64, 128)


def demands(path, pattern):
    selected = set()
    unsupported = set()
    matcher = re.compile(pattern) if pattern else None
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.startswith("TUNEDUMP_GEMV "):
                continue
            fields = line.split()
            if len(fields) != 7:
                raise ValueError(f"{path}:{line_number}: malformed TUNEDUMP_GEMV line")
            _, m, n, k, quant, opcode, state = fields
            scope = " ".join(fields[1:6])
            if matcher and not matcher.search(scope):
                continue
            if state not in ("HIT", "MISS"):
                raise ValueError(f"{path}:{line_number}: invalid census state {state}")
            try:
                dims = (int(m), int(n), int(k))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: nonnumeric census dimensions") from error
            mapping = SUPPORTED.get(opcode)
            if mapping is None:
                unsupported.add(opcode)
                continue
            family, expected_quant = mapping
            if quant != expected_quant:
                raise ValueError(
                    f"{path}:{line_number}: {opcode} says {quant}, expected {expected_quant}"
                )
            selected.add((*dims, quant, family))
    if unsupported:
        raise ValueError(
            "selected census contains unsupported GEMV opcodes: "
            + ", ".join(sorted(unsupported))
            + "; narrow --filter explicitly or extend the harness"
        )
    if not selected:
        raise ValueError("selected census contains no supported GEMV demand")
    return selected


def sample_identity(row):
    match = re.fullmatch(r"(.+)_m(1|2|4|8|16)", row.get("sym", ""))
    if not match or match.group(1) not in SYMBOL_FAMILY:
        raise ValueError(f"unexpected sweep symbol {row.get('sym')!r}")
    family, quant = SYMBOL_FAMILY[match.group(1)]
    if row.get("quant") != quant:
        raise ValueError(f"{row.get('sym')}: symbol/quant mismatch")
    dims = tuple(row.get(key) for key in ("m", "n", "k"))
    if any(type(value) is not int or value <= 0 for value in dims):
        raise ValueError(f"{row.get('sym')}: invalid sample dimensions")
    if dims[0] not in DEMAND_MMS:
        raise ValueError(f"{row.get('sym')}: unexpected demand M={dims[0]}")
    bucket = int(match.group(2))
    if row.get("mm") != bucket:
        raise ValueError(f"{row.get('sym')}: symbol/bucket mismatch")
    key = (*dims, quant, family)
    return key, (key, bucket, row["sym"])


def expected_samples(wanted, obj_mm):
    if obj_mm not in BUCKETS:
        raise ValueError(f"OBJ_MM must be one of {BUCKETS}, got {obj_mm}")
    expected = set()
    coverage = set()
    for key in wanted:
        m, n, k, quant, family = key
        stem = FAMILY_SYMBOL[(family, quant)]
        if quant == "None":
            for runtime_m in DEMAND_MMS:
                expanded = (runtime_m, n, k, quant, family)
                coverage.add(expanded)
                for bucket in BUCKETS:
                    expected.add((expanded, bucket, f"{stem}_m{bucket}"))
        elif quant == "Mxfp4":
            if m > obj_mm:
                raise ValueError(
                    f"Mxfp4 demand M={m} exceeds the compiled OBJ_MM={obj_mm}"
                )
            coverage.add(key)
            expected.add((key, obj_mm, f"{stem}_m{obj_mm}"))
        else:
            raise ValueError(f"no coverage rule for quant {quant}")
    return expected, coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "filter"))
    parser.add_argument("--census", required=True)
    parser.add_argument("--filter", default="")
    parser.add_argument("--raw")
    parser.add_argument("--output")
    parser.add_argument("--obj-mm", type=int, default=16)
    args = parser.parse_args()
    wanted = demands(args.census, args.filter)
    expected, coverage = expected_samples(wanted, args.obj_mm)
    if args.mode == "plan":
        shapes = {}
        for _, n, k, quant, family in wanted:
            shapes.setdefault((n, k), set()).add(FAMILY_SYMBOL[(family, quant)])
        for (n, k), arms in sorted(shapes.items()):
            print(f"{n}\t{k}\tcensus-n{n}-k{k}\t{','.join(sorted(arms))}")
        return
    if not args.raw or not args.output:
        parser.error("filter needs --raw and --output")
    kept = []
    seen = set()
    planned_shapes = {(key[1], key[2]) for key in wanted}
    with open(args.raw) as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{args.raw}:{line_number}: malformed JSON") from error
            key, identity = sample_identity(row)
            if identity in seen:
                raise ValueError(f"{row.get('sym')}: duplicate sweep sample")
            seen.add(identity)
            if key not in coverage:
                if (key[1], key[2]) not in planned_shapes:
                    raise ValueError(f"{row.get('sym')}: unexpected sweep shape")
                continue
            if identity not in expected:
                raise ValueError(f"{row.get('sym')}: unexpected demanded sample")
            if row.get("correct") is not True:
                raise ValueError(f"{row.get('sym')}: demanded sample failed correctness")
            samples = row.get("samples_ns")
            if not isinstance(samples, list) or len(samples) < 5 or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                for value in samples
            ):
                raise ValueError(f"{row.get('sym')}: invalid or insufficient timing samples")
            kept.append(row)
    missing = expected - seen
    if missing:
        raise ValueError(f"sweep missed {len(missing)} demanded rung sample(s)")
    with open(args.output, "x") as output:
        for row in kept:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"kept {len(kept)} passing rows covering {len(coverage)} runtime census cases")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, re.error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
