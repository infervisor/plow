#!/usr/bin/env python3
import argparse
import array
import glob
import json
import math
import os
import struct
import sys


def fail(message):
    raise ValueError(message)


def read_bf16(path):
    if os.path.getsize(path) == 0 or os.path.getsize(path) % 2:
        fail(f"invalid bf16 snapshot size: {path}")
    raw = array.array("H")
    with open(path, "rb") as source:
        raw.fromfile(source, os.path.getsize(path) // 2)
    if sys.byteorder != "little":
        raw.byteswap()
    return [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in raw]


def load_run(report_path, snap_dir, asset, packet, objects, checkpoint, prompt, steps, width, tp,
             require_snapshot_argmax):
    with open(report_path) as source:
        report = json.load(source)
    needed_outputs = steps + 1
    if report.get("schema") != "plowrt.bench.v1" or report.get("vendor") != "Some(Amd)":
        fail(f"tp={tp}: not an AMD production bench report")
    if (report.get("num_gpus"), report.get("parallel")) != (tp, "tp"):
        fail(f"tp={tp}: wrong TP engine identity")
    if (report.get("requests"), report.get("completed"), report.get("failed")) != (1, 1, 0):
        fail(f"tp={tp}: request did not complete exactly once")
    if report.get("warmup_requests") != 0:
        fail(f"tp={tp}: unexpected warmup request")
    if report.get("prompt_tokens") != len(prompt) or report.get("output_tokens") != needed_outputs:
        fail(f"tp={tp}: prompt/output counts changed")
    scheduler = report.get("scheduler") or {}
    if scheduler.get("rejected") != 0 or scheduler.get("admit_shed") != 0:
        fail(f"tp={tp}: scheduler rejected or shed work")
    audit = report.get("token_audit") or {}
    outputs = audit.get("output_token_ids")
    if audit.get("prompt_token_ids") != [prompt]:
        fail(f"tp={tp}: exact prompt changed")
    if not isinstance(outputs, list) or len(outputs) != 1 or len(outputs[0]) != needed_outputs:
        fail(f"tp={tp}: incomplete token stream")
    if not any(outputs[0]):
        fail(f"tp={tp}: all-zero token stream")

    real = os.path.realpath
    artifacts = report.get("artifacts") or {}
    packet_identity = artifacts.get("packet") or {}
    if real(report.get("asset_dir", "")) != real(asset):
        fail(f"tp={tp}: wrong asset directory identity")
    if real(packet_identity.get("path", "")) != real(packet):
        fail(f"tp={tp}: wrong packet identity")
    if not packet_identity.get("checksum"):
        fail(f"tp={tp}: missing packet checksum")
    build = artifacts.get("build_manifest") or {}
    if real(build.get("path", "")) != real(os.path.join(asset, "build.json")):
        fail(f"tp={tp}: wrong build manifest identity")
    if not build.get("checksum"):
        fail(f"tp={tp}: missing build manifest checksum")
    checkpoint_identity = artifacts.get("checkpoint") or {}
    if real(checkpoint_identity.get("path", "")) != real(checkpoint):
        fail(f"tp={tp}: wrong checkpoint identity")
    if not checkpoint_identity.get("layout_checksum"):
        fail(f"tp={tp}: missing checkpoint layout checksum")
    inventory = artifacts.get("object_inventory") or []
    object_root = real(objects)
    if not inventory or not all(
        os.path.commonpath([real(item.get("path", "")), object_root]) == object_root
        for item in inventory
    ):
        fail(f"tp={tp}: wrong object inventory")
    if any(not item.get("checksum") for item in inventory):
        fail(f"tp={tp}: missing object checksum")
    if (report.get("engine") or {}).get("batch_capacity") != 1:
        fail(f"tp={tp}: act.logits is not a single-row vector")

    diagnostics = report.get("diagnostics") or {}
    if diagnostics.get("supported") is not True or diagnostics.get("complete") is not True or diagnostics.get("overflowed") is not False:
        fail(f"tp={tp}: missing or partial diagnostics")
    agreement = diagnostics.get("rank_agreement")
    if tp == 1:
        if agreement is not None:
            fail("tp=1: unexpected rank agreement policy")
    elif not isinstance(agreement, dict) or (
        agreement.get("ranks") != tp
        or agreement.get("sampled_token_every") != 1
        or agreement.get("counter_audit_every_dispatch") is not True
        or agreement.get("prefill_completion_all_ranks") is not True
    ):
        fail(f"tp={tp}: incomplete every-token rank agreement policy")

    ticks = len(prompt) + steps
    paths = []
    for tick in range(ticks):
        matches = glob.glob(os.path.join(snap_dir, f"t{tick:05d}_r*_act_logits.bin"))
        if len(matches) != 1:
            fail(f"tp={tp}: snapshot tick {tick} has {len(matches)} act.logits files")
        paths.append(matches[0])
    if len(glob.glob(os.path.join(snap_dir, "t*_act_logits.bin"))) != ticks:
        fail(f"tp={tp}: unexpected snapshot count")
    selected = [paths[len(prompt) - 1], *paths[len(prompt):]]
    vectors = [read_bf16(path) for path in selected]
    if any(len(row) != width for row in vectors):
        fail(f"tp={tp}: snapshot is not the expected {width}-entry logit vector")
    if require_snapshot_argmax and any(
        max(range(len(row)), key=row.__getitem__) != token
        for row, token in zip(vectors, outputs[0])
    ):
        fail(f"tp={tp}: token audit does not match snapshot argmax")
    return vectors, outputs[0]


def main():
    parser = argparse.ArgumentParser()
    for name in ["report1", "snap1", "asset1", "packet1", "report8", "snap8", "asset8", "packet8", "objects", "checkpoint", "prompt"]:
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--cos", required=True, type=float)
    parser.add_argument("--layers", required=True, type=int)
    parser.add_argument("--vocab", required=True, type=int)
    parser.add_argument("--sharded8", action="store_true")
    args = parser.parse_args()
    prompt = [int(token.strip()) for token in args.prompt.split(",")]
    one, tokens1 = load_run(
        args.report1, args.snap1, args.asset1, args.packet1, args.objects, args.checkpoint,
        prompt, args.steps, args.vocab, 1, True)
    width8 = args.vocab // 8 if args.sharded8 else args.vocab
    if args.sharded8 and args.vocab % 8:
        fail(f"vocab {args.vocab} is not divisible by tp=8")
    eight, tokens8 = load_run(
        args.report8, args.snap8, args.asset8, args.packet8, args.objects, args.checkpoint,
        prompt, args.steps, width8, 8, not args.sharded8)
    if tokens1 != tokens8:
        first = next(i for i, (a, b) in enumerate(zip(tokens1, tokens8)) if a != b)
        fail(
            f"global greedy stream differs at output {first}: "
            f"full-vocab={tokens1[first]} sharded={tokens8[first]}"
        )
    bad = False
    tags = ["prefill", *[f"{i:03d}" for i in range(args.steps)]]
    for tag, x, y in zip(tags, one, eight):
        if args.sharded8:
            x = x[:width8]
        if len(x) != len(y):
            print(f"  N={args.layers} {tag:8s} SHAPE {len(x)} vs {len(y)}")
            bad = True
            continue
        dot = sum(a * b for a, b in zip(x, y))
        nx = math.sqrt(sum(a * a for a in x))
        ny = math.sqrt(sum(b * b for b in y))
        cosine = dot / (nx * ny)
        a1 = max(range(len(x)), key=x.__getitem__)
        a8 = max(range(len(y)), key=y.__getitem__)
        max_abs = max(abs(a - b) for a, b in zip(x, y))
        ok = cosine >= args.cos and a1 == a8
        print(f"  N={args.layers} {tag:8s} cos {cosine:.8f}  argmax {a1} vs {a8}  maxabs {max_abs:.5f}  {'ok' if ok else 'MISMATCH'}")
        bad |= not ok
    if args.sharded8 and not bad:
        outside = sum(token >= width8 for token in tokens8)
        print(
            f"  global greedy {len(tokens8)}/{len(tokens8)} exact; "
            f"{outside} winner(s) outside rank 0's vocab shard"
        )
    if bad:
        if args.layers == 1:
            print("  => N=1 localises disagreement to the KDA mixer, dense FFN, attention all-reduce, or weight sharding.")
        else:
            print("  => N>=2 disagreeing while N=1 agrees points at the latent MoE half or its peer slots.")
    return int(bad)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
