#!/usr/bin/env python3
"""vLLM decode step time per batch rung for a model (same shape as plowrt bench rung sweeps):
B requests of 64 random prompt tokens, exactly OUT output tokens each (ignore_eos), greedy.
step_ms = (wall(OUT) - wall(1)) / (OUT - 1), prefill excluded by differencing."""
import argparse, time
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", type=int, default=384)
ap.add_argument("--concs", default="1,2,4,8,16,32,64")
args = ap.parse_args()
llm = LLM(args.model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.6,
          enable_prefix_caching=False, max_num_seqs=64)
import random
random.seed(0)
for b in map(int, args.concs.split(",")):
    prompts = [{"prompt_token_ids": [random.randrange(1000, 100000) for _ in range(64)]} for _ in range(b)]
    def run(n):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        t = time.perf_counter(); llm.generate(prompts, sp, use_tqdm=False); return time.perf_counter() - t
    run(8)
    t1 = min(run(1) for _ in range(2)); tn = min(run(args.out) for _ in range(2))
    print(f"vllm c={b} step_ms={(tn - t1) / (args.out - 1) * 1e3:.3f}")
