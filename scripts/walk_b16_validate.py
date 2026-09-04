#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import sys


def fail(message):
    raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--measured", required=True, type=int)
    args = parser.parse_args()
    with open(args.report) as source:
        report = json.load(source)
    with open(args.prompts) as source:
        prompts = [[int(token.strip()) for token in row.split(",")]
                   for row in source.read().splitlines()]
    outputs_expected = args.warmup + args.measured + 1
    if report.get("schema") != "plowrt.bench.v1" or report.get("vendor") != "Some(Amd)":
        fail("not an AMD production bench report")
    if report.get("num_gpus") != 1 or report.get("parallel") != "tp":
        fail("walk gate requires the single-GPU production engine")
    if (report.get("requests"), report.get("completed"), report.get("failed")) != (args.batch, args.batch, 0):
        fail("compiled-width request set did not complete")
    if report.get("warmup_requests") != 0:
        fail("request-level warmups would change the four-dispatch warmup")
    if report.get("prompt_tokens") != sum(map(len, prompts)):
        fail("prompt token count changed")
    if report.get("output_tokens") != args.batch * outputs_expected:
        fail("output token count changed")
    scheduler = report.get("scheduler") or {}
    if scheduler.get("rejected") != 0 or scheduler.get("admit_shed") != 0:
        fail("scheduler rejected or shed work")
    if (report.get("engine") or {}).get("batch_capacity") != args.batch:
        fail("loaded engine width differs from the arm")

    audit = report.get("token_audit") or {}
    outputs = audit.get("output_token_ids")
    if audit.get("prompt_token_ids") != prompts:
        fail("token audit changed prompt order")
    if not isinstance(outputs, list) or len(outputs) != args.batch or any(
        not isinstance(row, list) or len(row) != outputs_expected for row in outputs
    ):
        fail("token audit returned incomplete per-slot streams")
    if not any(token for row in outputs for token in row):
        fail("all-zero streams make identity vacuous")
    for slot in range(2, args.batch):
        if outputs[slot] != outputs[slot % 2]:
            fail(f"slot {slot} differs from slot {slot % 2} with the same prompt")

    real = os.path.realpath
    artifacts = report.get("artifacts") or {}
    packet = artifacts.get("packet") or {}
    build = artifacts.get("build_manifest") or {}
    weights = artifacts.get("weights_manifest") or {}
    checkpoint = artifacts.get("checkpoint") or {}
    if real(report.get("asset_dir", "")) != real(args.assets):
        fail("wrong asset directory")
    for identity, expected, label in [
        (packet, os.path.join(args.assets, "model.pkt"), "packet"),
        (build, os.path.join(args.assets, "build.json"), "build manifest"),
        (weights, os.path.join(args.assets, "weights.json"), "weights manifest"),
    ]:
        if real(identity.get("path", "")) != real(expected) or not identity.get("checksum"):
            fail(f"wrong or unhashed {label}")
    if real(checkpoint.get("path", "")) != real(args.checkpoint) or not checkpoint.get("layout_checksum"):
        fail("wrong or unhashed checkpoint")
    inventory = artifacts.get("object_inventory") or []
    object_root = real(args.objects)
    if not inventory or not all(
        os.path.commonpath([real(item.get("path", "")), object_root]) == object_root
        and item.get("checksum") for item in inventory
    ):
        fail("wrong or unhashed object inventory")

    diagnostics = report.get("diagnostics") or {}
    if diagnostics.get("supported") is not True or diagnostics.get("complete") is not True or diagnostics.get("overflowed") is not False:
        fail("missing or partial engine diagnostics")
    selections = diagnostics.get("decode_selections") or []
    needed = args.warmup + args.measured
    if len(selections) < needed:
        fail("too few decode dispatch timings")
    selected = selections[-needed:]
    if any(row.get("occupied_rows") != args.batch or row.get("bucket") != args.batch
           or row.get("steps") != 1 or not isinstance(row.get("elapsed_ns"), int)
           or row["elapsed_ns"] <= 0 for row in selected):
        fail("timed window contains a partial, multistep, or invalid dispatch")
    measured = [row["elapsed_ns"] for row in selected[args.warmup:]]
    if len(measured) != args.measured:
        fail("measured dispatch count changed")
    mean_ms = statistics.fmean(measured) / 1e6
    print(f"  {args.measured} dispatches x batch {args.batch} after {args.warmup} warmups:")
    print(f"  tpot {mean_ms:.3f} ms  |  aggregate {args.batch * 1e3 / mean_ms:.1f} tok/s")
    print(f"  {args.batch} complete per-slot streams agree within their prompt classes")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
