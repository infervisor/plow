#!/usr/bin/env python3
"""Create matched finite fixtures for K3 T8 verification and serial replay."""

import argparse
import array
import json
import os
import re
import subprocess
from pathlib import Path


KDA_RE = re.compile(r"^kv\.\d+\.(state|conv_state\.[qkv])$")
MLA_RE = re.compile(r"^kv\.\d+\.(ckv|krot|scale)$")
JOURNAL_BANKS = 9


def tensor_table(plowrt: Path, blob: Path) -> dict[str, int]:
    env = dict(os.environ)
    env["RUST_LOG"] = "error"
    raw = subprocess.check_output(
        [
            str(plowrt),
            "disasm",
            str(blob),
            "--program",
            "1",
            "--tensors",
            "--no-analysis",
            "--format",
            "json",
        ],
        env=env,
    )
    return {tensor["name"]: int(tensor["bytes"]) for tensor in json.loads(raw)["tensors"]}


def patterned_f32(nbytes: int, seed: int) -> bytes:
    if nbytes % 4:
        raise SystemExit(f"f32 fixture has non-word byte count {nbytes}")
    values = (2**-10, -(2**-11), 2**-12, -(2**-13), 2**-14, -(2**-15))
    count = nbytes // 4
    return array.array("f", (values[(seed + i) % len(values)] for i in range(count))).tobytes()


def patterned_bf16(nbytes: int, seed: int) -> bytes:
    if nbytes % 2:
        raise SystemExit(f"bf16 fixture has odd byte count {nbytes}")
    values = (0x3C00, 0xBC00, 0x3C80, 0xBC80, 0x3D00, 0xBD00)
    count = nbytes // 2
    return array.array("H", (values[(seed + i) % len(values)] for i in range(count))).tobytes()


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"{path}\t{len(data)}")


def check_matching_names(verifier: dict[str, int], serial: dict[str, int], pattern: re.Pattern) -> list[str]:
    verifier_names = {name for name in verifier if pattern.fullmatch(name)}
    serial_names = {name for name in serial if pattern.fullmatch(name)}
    if verifier_names != serial_names:
        only_v = sorted(verifier_names - serial_names)
        only_s = sorted(serial_names - verifier_names)
        raise SystemExit(f"fixture tensor sets differ: verifier-only={only_v}, serial-only={only_s}")
    return sorted(verifier_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-blob", type=Path, required=True)
    parser.add_argument("--serial-blob", type=Path, required=True)
    parser.add_argument("--verifier-out", type=Path, required=True)
    parser.add_argument("--serial-out", type=Path, required=True)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--plowrt", type=Path, default=Path("target/release/plowrt"))
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    if args.history <= 0:
        raise SystemExit("--history must be positive")
    verifier = tensor_table(args.plowrt, args.verifier_blob)
    serial = tensor_table(args.plowrt, args.serial_blob)

    if verifier.get("kv.spec_commit") != 4 or "kv.spec_commit" in serial:
        raise SystemExit("expected one verifier-only u32 kv.spec_commit tensor")
    if verifier.get("act.x") != 8 * serial.get("act.x", 0):
        raise SystemExit("verifier act.x must contain exactly eight serial rows")

    row = patterned_bf16(serial["act.x"], args.seed)
    write(args.serial_out / "act.x.bin", row)
    write(args.verifier_out / "act.x.bin", row * 8)
    write(args.verifier_out / "kv.spec_commit.bin", bytes(4))

    for ordinal, name in enumerate(check_matching_names(verifier, serial, KDA_RE)):
        base_bytes = serial[name]
        if verifier[name] != JOURNAL_BANKS * base_bytes:
            raise SystemExit(
                f"{name}: verifier has {verifier[name]} B, expected {JOURNAL_BANKS} x {base_bytes} B"
            )
        base = patterned_f32(base_bytes, args.seed + ordinal + 1)
        write(args.serial_out / f"{name}.bin", base)
        write(args.verifier_out / f"{name}.bin", base + bytes((JOURNAL_BANKS - 1) * base_bytes))

    mla_names = check_matching_names(verifier, serial, MLA_RE)
    layers = sorted({name.split(".")[1] for name in mla_names}, key=int)
    for ordinal, layer in enumerate(layers):
        names = {kind: f"kv.{layer}.{kind}" for kind in ("ckv", "krot", "scale")}
        if any(name not in serial for name in names.values()):
            raise SystemExit(f"MLA layer {layer} is missing ckv, krot, or scale")
        if any(verifier[name] != serial[name] for name in names.values()):
            raise SystemExit(f"MLA layer {layer} differs in verifier and serial allocation")

        ctx_bytes = serial[names["scale"]]
        if ctx_bytes % 4:
            raise SystemExit(f"{names['scale']}: scale allocation is not f32")
        ctx = ctx_bytes // 4
        if args.history > ctx:
            raise SystemExit(f"--history {args.history} exceeds MLA context capacity {ctx}")
        ckv_row = serial[names["ckv"]] // ctx
        krot_row = serial[names["krot"]] // ctx
        if ckv_row * ctx != serial[names["ckv"]] or krot_row * ctx != serial[names["krot"]]:
            raise SystemExit(f"MLA layer {layer} cache allocations do not divide by ctx={ctx}")

        fp8_values = (0x20, 0x28, 0x30, 0x38, 0xA0, 0xA8, 0xB0, 0xB8)
        live_ckv = bytes(
            fp8_values[(args.seed + ordinal + i) % len(fp8_values)]
            for i in range(args.history * ckv_row)
        )
        ckv = live_ckv + bytes((ctx - args.history) * ckv_row)
        live_krot = patterned_bf16(args.history * krot_row, args.seed + ordinal + 101)
        krot = live_krot + bytes((ctx - args.history) * krot_row)
        scale = array.array("f", [2**-5] * args.history + [0.0] * (ctx - args.history)).tobytes()

        for name, data in ((names["ckv"], ckv), (names["krot"], krot), (names["scale"], scale)):
            write(args.serial_out / f"{name}.bin", data)
            write(args.verifier_out / f"{name}.bin", data)


if __name__ == "__main__":
    main()
