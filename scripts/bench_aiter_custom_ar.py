#!/usr/bin/env python3
"""TP custom-allreduce probe for the vLLM-pinned AITER implementation."""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist


def parse_shape(text: str) -> tuple[int, int]:
    try:
        rows, hidden = (int(v) for v in text.lower().split("x", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected ROWSxHIDDEN, got {text!r}") from exc
    if rows <= 0 or hidden <= 0 or rows * hidden * 2 % 16:
        raise argparse.ArgumentTypeError("shape must be positive and BF16 bytes divisible by 16")
    return rows, hidden


def dispatch(rows: int, hidden: int, world: int) -> dict:
    elements = rows * hidden
    byte_count = elements * 2
    packs = elements // 8
    if byte_count < (80 * 1024 if world <= 8 else 160 * 1024):
        lanes_per_rank = 512 // world
        blocks = min(80, (packs + lanes_per_rank - 1) // lanes_per_rank)
        algorithm = "one_stage"
    else:
        blocks = min(80, (packs // world + 512 // world - 1) // (512 // world))
        algorithm = "two_stage_rs_ag"
    return {
        "algorithm": algorithm,
        "blocks": blocks,
        "threads": 512,
        "bytes_per_rank": byte_count,
    }


def elapsed_ms(call, warmup: int, samples: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(samples):
        call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / samples


def rank_summary(value: float) -> dict:
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return {
        "rank_ms": values,
        "max_rank_ms": max(values),
        "median_rank_ms": statistics.median(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        dest="shapes",
        help="ROWSxHIDDEN; repeatable",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--oracle", choices=("order", "benign"), default="order")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 1:
        parser.error("warmup and samples must be positive")
    shapes = args.shapes or [
        (1, 3584),
        (1, 7168),
        (4096, 3584),
        (4096, 7168),
        (8192, 3584),
        (8192, 7168),
    ]

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 8:
        raise RuntimeError(f"this parity harness requires TP8, got world_size={world}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    largest = max(rows * hidden * 2 for rows, hidden in shapes)
    communicator = CustomAllreduce(
        dist.group.WORLD,
        local_rank,
        max_size=max(128 * 1024 * 1024, largest),
        enable_register_for_capturing=True,
    )
    if communicator.disabled:
        raise RuntimeError("AITER custom allreduce is disabled")

    results = []
    for rows, hidden in shapes:
        if args.oracle == "order":
            rank_values = [16777216.0, 1.0, -16777216.0] + [0.0] * (world - 3)
            expected = 0.0
        else:
            rank_values = [float(r + 1) for r in range(world)]
            expected = float(world * (world + 1) // 2)
        inp = torch.full(
            (rows, hidden), rank_values[rank], dtype=torch.bfloat16, device="cuda"
        )
        out = torch.empty_like(inp)
        communicator.register_input_buffer(inp)
        dist.barrier()

        def registered_call():
            communicator.all_reduce(inp, out=out, registered_input=True)

        registered_ms = rank_summary(elapsed_ms(registered_call, args.warmup, args.samples))
        bad = int(torch.count_nonzero(out != expected).item())
        bad_by_rank = [None] * world
        dist.all_gather_object(bad_by_rank, bad)

        dist.barrier()

        def eager_call():
            communicator.all_reduce(inp, out=out, registered_input=False)

        eager_ms = rank_summary(elapsed_ms(eager_call, args.warmup, args.samples))
        item = {
            "rows": rows,
            "hidden": hidden,
            **dispatch(rows, hidden, world),
            "registered": registered_ms,
            "eager_copy_in": eager_ms,
            "bad_elements_by_rank": bad_by_rank,
            "oracle": args.oracle,
            "rank_values": rank_values,
            "rank_order_expected": expected,
        }
        results.append(item)
        if rank == 0:
            print(json.dumps(item, sort_keys=True), flush=True)
        del inp, out

    if rank == 0:
        report = {
            "backend": "AITER_CUSTOM",
            "world_size": world,
            "dtype": "bfloat16",
            "warmup": args.warmup,
            "samples": args.samples,
            "oracle": args.oracle,
            "aiter_version": "0.1.19",
            "results": results,
        }
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
