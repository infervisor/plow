"""Time DeepSeek-V4.1-Flash's SHIPPED reference at a fixed prefill length.

Run under torchrun with `PYTHONPATH` pointing at the checkpoint's `inference/`
directory; see scripts/dsv41_reference_8k.sh.

The reference's own `generate()` reports no timing, and it is the Stage-5
oracle, so it is imported rather than edited. This driver reproduces its load
sequence exactly (`generate.py:108-133`) and times the two phases separately:

  TTFT  one forward over the whole prompt at prev_pos=0  -- the prefill
  TPOT  the mean of `--decode-steps` single-token forwards after it

Numbers from here are a CEILING, not a target: eager PyTorch, tilelang
kernels, batch 1, no CUDA graphs, no paged KV. The first call of every kernel
shape also pays tilelang JIT, which is why `--warmup-len` runs first and is
discarded.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from transformers import AutoTokenizer


def _unnix_ld_path() -> None:
    """Drop the nix entries from LD_LIBRARY_PATH now that torch is imported.

    The vllm-python wrapper puts nix glibc/gcc first so `import torch` resolves;
    those libraries are mapped by the time this runs. But tilelang JIT-compiles
    its HIP kernels at the first call of every kernel shape and SHELLS OUT to do
    it, and the child `sh` then loads the same nix glibc against the system
    loader:

        sh: symbol lookup error: .../glibc-2.42-67/lib/libc.so.6:
            undefined symbol: __tunable_is_initialized, version GLIBC_PRIVATE

    which fails every kernel compile and so every forward. Keep the ROCm entry
    -- torch still dlopens from it -- and drop only /nix/store.
    """
    parts = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    kept = [p for p in parts if p and not p.startswith("/nix/store")]
    os.environ["LD_LIBRARY_PATH"] = ":".join(kept)

from model import ModelArgs, Transformer  # from the checkpoint's inference/
from generate import load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--prefill-len", type=int, default=8192)
    p.add_argument("--warmup-len", type=int, default=256)
    p.add_argument("--decode-steps", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=16384)
    p.add_argument("--json-out", default="")
    args_cli = p.parse_args()
    _unnix_ld_path()

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    say = print if rank == 0 else (lambda *_, **__: None)

    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)

    with open(args_cli.config) as f:
        margs = ModelArgs(**json.load(f))
    # Runtime limits, not model shape: the config ships neither, and the
    # default max_seq_len (4096) cannot hold an 8k prompt.
    margs.max_batch_size = 1
    margs.max_seq_len = args_cli.max_seq_len
    margs.temperature = 0.0

    tokenizer = AutoTokenizer.from_pretrained(args_cli.ckpt_path)
    say("build model")
    with torch.device("cuda"):
        model = Transformer(margs, tokenizer)
    say("load model")
    load_model(model, os.path.join(args_cli.ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    torch.set_default_device("cuda")

    with open(args_cli.prompt_file) as f:
        text = f.read().strip()
    ids = tokenizer.encode(text)
    assert len(ids) >= args_cli.prefill_len, (
        f"prompt has {len(ids)} tokens, need {args_cli.prefill_len}"
    )
    ids = ids[: args_cli.prefill_len]
    say(f"prompt: {len(ids)} tokens, world_size={world_size}")

    def forward(tok_slice, prev_pos):
        return model.forward(tok_slice, prev_pos)[0]

    total = args_cli.prefill_len + args_cli.decode_steps + 1
    tokens = torch.full((1, total), -1, dtype=torch.long, device="cuda")
    tokens[0, : len(ids)] = torch.tensor(ids, dtype=torch.long, device="cuda")

    # --- warmup: same code path, shorter, so tilelang JIT is not in the number
    with torch.inference_mode():
        forward(tokens[:, : args_cli.warmup_len], 0)
    torch.cuda.synchronize()
    if world_size > 1:
        dist.barrier()

    # The warmup wrote the KV caches at positions 0..warmup_len; rebuild the
    # model state by re-running from prev_pos=0, which is what a fresh request
    # does. The caches are overwritten in place, not appended, so this is the
    # same work the first request of a served sequence performs.
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nxt = forward(tokens[:, : args_cli.prefill_len], 0)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        tokens[0, args_cli.prefill_len] = nxt[0]
        steps = []
        prev = args_cli.prefill_len
        for i in range(args_cli.decode_steps):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            nxt = forward(tokens[:, prev : prev + 1], prev)
            torch.cuda.synchronize()
            steps.append(time.perf_counter() - t1)
            tokens[0, prev + 1] = nxt[0]
            prev += 1

    if rank == 0:
        tpot = statistics.mean(steps)
        result = {
            "model": "DeepSeek-V4.1-Flash",
            "runtime": "shipped reference (eager torch + tilelang)",
            "world_size": world_size,
            "prefill_tokens": args_cli.prefill_len,
            "ttft_s": round(ttft, 4),
            "prefill_tok_per_s": round(args_cli.prefill_len / ttft, 1),
            "decode_steps": args_cli.decode_steps,
            "tpot_ms_mean": round(tpot * 1e3, 2),
            "tpot_ms_min": round(min(steps) * 1e3, 2),
            "target_ttft_s": 0.300,
            "over_target_x": round(ttft / 0.300, 2),
        }
        say(json.dumps(result, indent=2))
        if args_cli.json_out:
            with open(args_cli.json_out, "w") as f:
                json.dump(result, f, indent=2)
            say(f"wrote {args_cli.json_out}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
