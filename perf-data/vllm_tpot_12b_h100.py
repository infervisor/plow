"""vLLM TPOT/TTFT reference for Gemma-4-12B-it on H100 NVL.

Same method as vllm_tpot_h100.py (26B) but parameterised on the model dir, since
the 12B checkpoint is not under /workspace/models (see
perf-data/gemma12b-h100-landing.md — it is fetched to /dev/shm).

  usage: vllm_tpot_12b_h100.py <ctx> [model_dir] [dtype]

TPOT is the slope (t129 - t1)/128 so the prefill is differenced out; t1 is the
1-token latency, i.e. TTFT for a ctx-token prompt.
"""
import os, time, sys
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams

ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
mp = sys.argv[2] if len(sys.argv) > 2 else "/dev/shm/models/gemma-4-12B-it"
dtype = sys.argv[3] if len(sys.argv) > 3 else "bfloat16"

llm = LLM(model=mp, dtype=dtype, max_model_len=ctx + 512,
          gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.90")),
          enforce_eager=False, tensor_parallel_size=1, trust_remote_code=True)
tok = llm.get_tokenizer()
# exact-length prompt in TOKENS
ids = (tok.encode("The history of science and technology is long. ") * 400)[:ctx]
prompt = tok.decode(ids)


def run(n, reps=3):
    sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
    llm.generate([prompt], sp)  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        o = llm.generate([prompt], sp)
        ts.append(time.perf_counter() - t0)
        assert len(o[0].outputs[0].token_ids) == n, len(o[0].outputs[0].token_ids)
    ts.sort()
    return ts[len(ts) // 2]


t1 = run(1)
t129 = run(129)
print(f"RESULT model={mp} dtype={dtype} ctx={ctx} t1={t1*1e3:.2f}ms "
      f"t129={t129*1e3:.2f}ms TPOT={(t129-t1)/128*1e3:.3f} ms/token")
