import json
for m in ("E2B", "E4B"):
    t = json.load(open(f"/tmp/models/google/gemma-4-{m}-it/config.json"))["text_config"]
    lt, sh = t["layer_types"], t["num_kv_shared_layers"]
    own = lt[: len(lt) - sh]
    fo, so = own.count("full_attention"), own.count("sliding_attention")
    fr, sr = lt.count("full_attention"), lt.count("sliding_attention")
    kvh, hd, ghd, win = t["num_key_value_heads"], t["head_dim"], t["global_head_dim"], t["sliding_window"]
    full_tok = 2 * kvh * ghd * 2
    slide = 2 * kvh * hd * 2 * win
    print(f"{m}: own full {fo} sliding {so}; readers full {fr} sliding {sr}; full KV/token/layer {full_tok} B; sliding ring {slide/2**20:.2f} MiB")
    print("| ctx | full layer KV MiB | allocated per seq MiB | read per token per seq MiB | c4 alloc GiB | c16 alloc GiB |")
    for c in (2048, 8192, 16384, 32768, 65536, 131072):
        fl = c * full_tok / 2**20
        alloc = fo * fl + so * slide / 2**20
        rd = fr * fl + sr * slide / 2**20
        print(f"| {c//1024}K | {fl:.0f} | {alloc:.0f} | {rd:.0f} | {4*alloc/1024:.2f} | {16*alloc/1024:.2f} |")
