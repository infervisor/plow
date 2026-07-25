import os, time, sys
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
from vllm import LLM, SamplingParams
mp = "/workspace/models/gemma-4-26B-A4B-it"
ctx = int(sys.argv[1]) if len(sys.argv)>1 else 1024
llm = LLM(model=mp, dtype="bfloat16", max_model_len=ctx+512, gpu_memory_utilization=0.90,
          enforce_eager=False, tensor_parallel_size=1, trust_remote_code=True)
tok = llm.get_tokenizer()
# exact-length prompt in TOKENS
ids = (tok.encode("The history of science and technology is long. ")*400)[:ctx]
prompt = tok.decode(ids)
def run(n, reps=3):
    sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
    llm.generate([prompt], sp)  # warm
    ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); o=llm.generate([prompt], sp); ts.append(time.perf_counter()-t0)
        assert len(o[0].outputs[0].token_ids)==n, len(o[0].outputs[0].token_ids)
    ts.sort(); return ts[len(ts)//2]
t1 = run(1); t129 = run(129)
print(f"RESULT ctx={ctx} t1={t1*1e3:.2f}ms t129={t129*1e3:.2f}ms TPOT={(t129-t1)/128*1e3:.3f} ms/token")
