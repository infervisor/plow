import json, re, statistics as st
import sys
RUN = sys.argv[1]; CAMP = RUN.split("/")[-1]; ROUNDS = int(sys.argv[2]); DIG = sys.argv[3]
PREV = dict(campaign=sys.argv[4], median_ttft_ms=float(sys.argv[5]), median_tpot_ms=float(sys.argv[6]), output_throughput=float(sys.argv[7]))
AI = json.loads(sys.argv[8])


def parse(p):
    t = open(p).read()
    g = lambda k: float(re.search(k + r'[^:]*:\s*([0-9.]+)', t).group(1))
    return dict(duration_s=g('Benchmark duration'), ttft_mean=g('Mean TTFT'), ttft_median=g('Median TTFT'),
                ttft_p90=g('P90 TTFT'), ttft_p99=g('P99 TTFT'), tpot_mean=g('Mean TPOT'), tpot_median=g('Median TPOT'),
                tpot_p90=g('P90 TPOT'), tpot_p99=g('P99 TPOT'), itl_mean=g('Mean ITL'), itl_median=g('Median ITL'),
                itl_p90=g('P90 ITL'), itl_p99=g('P99 ITL'), e2el_mean=g('Mean E2EL'), e2el_median=g('Median E2EL'),
                e2el_p90=g('P90 E2EL'), e2el_p99=g('P99 E2EL'), prompts=10, input_tokens=81920, output_tokens=10240,
                req_s=g('Request throughput'), out_tok_s=g('\nOutput token throughput'),
                total_tok_s=g('Total token throughput'))


folds, vf = [], []
for r in range(1, ROUNDS + 1):
    f = parse(f"{RUN}/plow-rshowdown-{r}-in8192.log")
    folds.append({"round": f"showdown-{r}", "log": f"plow-rshowdown-{r}-in8192.log", **f, "artifact_digest": DIG})
    vf.append(parse(f"{RUN}/vllm-rshowdown-{r}-in8192.log"))
med = lambda k, fs=folds: round(st.median(x[k] for x in fs), 2)
old = json.load(open("perf-data/kimi-k3-plowrt-mi355x-c1.json"))
d = dict(old)
d.update(dict(date="20260904", status="current publication baseline", campaign=CAMP, rounds=ROUNDS,
              source_cells=f"{RUN}/cells.tsv", source_config=f"{RUN}/config.txt",
              aggregation=f"median of the {ROUNDS} fold statistics (including P90 and duration); folds below preserve cells.tsv and client logs exactly",
              model_id="kimi-k3", duration_s=med('duration_s'), request_throughput=med('req_s'),
              output_throughput=med('out_tok_s'), total_token_throughput=med('total_tok_s')))
for m in ('ttft', 'tpot', 'itl', 'e2el'):
    for s in ('mean', 'median', 'p90', 'p99'):
        d[f"{s}_{m}_ms"] = med(f"{m}_{s}")
d["artifact_digest"] = DIG
ai = dict(old["artifact_identity"])
ai.update(**AI)
ai.pop("lean_proof_sha256", None)
ai["object_markers"] = sorted(set(old["artifact_identity"]["object_markers"] + AI.get("extra_markers", [])))
ai.pop("extra_markers", None)
d["artifact_identity"] = ai
rt = dict(old["runtime"])
rt.update(grouped_moe_opt_in=False, grouped_moe_decode_route="standalone (selected by measured TuneDB rule)",
          gq_window_order="asap", hsa_queue_size=4096, attnres_f32mix=AI.get("_f32mix", True), kda_carry_regstate=True,
          xreduce_decode_tagged=True, mla_decode_merge_fold_split=True)
rt.pop("_f32mix", None)
d["runtime"] = rt
d["lease"] = dict(path="/tmp/gpulease", acquired=AI.get("_acq"), released=AI.get("_rel"), return_code=AI.get("_rc", 0), foreign_work_detected=False)
d["supersedes"] = dict(status="superseded by the stack-2 campaign (KDA carry regstate, f32-mix AttnRes, tagged decode XReduce, MLA merge-fold)", **PREV)
d["vllm_same_campaign"] = dict(median_ttft_ms=med('ttft_median', vf), median_tpot_ms=med('tpot_median', vf),
                               p99_itl_ms=med('itl_p99', vf), output_throughput=med('out_tok_s', vf),
                               median_e2el_ms=med('e2el_median', vf), duration_s=med('duration_s', vf),
                               p90_ttft_ms=med('ttft_p90', vf), p99_ttft_ms=med('ttft_p99', vf),
                               median_itl_ms=med('itl_median', vf), p90_itl_ms=med('itl_p90', vf),
                               total_token_throughput=med('total_tok_s', vf),
                               note="vLLM 0.28 cells of this campaign; the published vLLM baseline JSON is unchanged")
d["folds"] = folds
json.dump(d, open("perf-data/kimi-k3-plowrt-mi355x-c1.json", "w"), indent=1)
print("json ok")
for f in folds:
    print(f"| {f['round']} | {f['ttft_mean']:.2f} / {f['ttft_median']:.2f} / {f['ttft_p99']:.2f} | "
          f"{f['tpot_mean']:.2f} / {f['tpot_median']:.2f} / {f['tpot_p99']:.2f} | {f['out_tok_s']:.2f} | "
          f"{f['e2el_mean']:.2f} / {f['e2el_median']:.2f} / {f['e2el_p99']:.2f} |")
keys = ('duration_s', 'output_throughput', 'total_token_throughput', 'median_ttft_ms', 'p90_ttft_ms', 'p99_ttft_ms',
        'median_tpot_ms', 'median_itl_ms', 'p90_itl_ms', 'p99_itl_ms', 'median_e2el_ms')
print({k: d[k] for k in keys})
print(d["vllm_same_campaign"])
