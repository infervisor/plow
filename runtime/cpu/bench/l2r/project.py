#!/usr/bin/env python3
"""project.py: per-stream batch-1 decode projection for L2-resident pipelined serving on 2-socket Xeon 6.
Inputs are the measurements on c8i.metal-48xl (/tmp/g4c/l2bw) plus stated assumptions; output is JSON
consumed by the plan page and a text table."""
import json, math, os

MODELS = "/tmp/models/google"
GB = 1e9

def cfg(m):
    c = json.load(open(os.path.join(MODELS, m, "config.json")))
    return c.get("text_config", c)

# body params (B) from safetensors headers (/tmp/g4c/wsize.py)
BODY = {"gemma-4-E2B-it": 1.88, "gemma-4-E4B-it": 4.00, "gemma-4-12B-it": 10.90, "gemma-4-31B-it": 29.29}

def gemma(m):
    t = cfg(m)
    lt = t["layer_types"]
    g = sum("full" in x for x in lt)
    kvh, hd, ghd = t["num_key_value_heads"], t["head_dim"], t["global_head_dim"]
    gkv = t.get("num_global_key_value_heads") or kvh
    return dict(name=m, layers=len(lt), glob=g, slid=len(lt) - g, body=BODY[m] * GB,
                lm=t["vocab_size"] * t["hidden_size"], window=t["sliding_window"],
                g_elem=gkv * ghd * (1 if t.get("attention_k_eq_v") else 2), s_elem=kvh * hd * 2)

LLAMA8 = dict(name="Llama-3.1-8B (dense reference)", layers=32, glob=32, slid=0, body=6.98 * GB, lm=0.525 * GB,
              window=0, g_elem=8 * 128 * 2, s_elem=0)

SOCKETS = {"aws": dict(label="AWS Xeon 6975P-C, 96 cores", cores=96),
           "gnr128": dict(label="Xeon 6980P, 128 cores", cores=128)}
SCEN = {"today": dict(label="stage code as measured", sync=2.6, points=5, l2=1.0, bw=95e9),
        "tuned": dict(label="tuned stage", sync=1.0, points=4, l2=1.5, bw=131e9)}
NET = {"rdma400": dict(label="400G RDMA, 2.5 us/hop", hop=2.5), "efa": dict(label="EFA-class, 12 us/hop", hop=12.0)}
BW_L2 = 131e9      # bytes/s per core, AMX-INT8 tile stream from L2 (measured 1 MiB/core, 96 cores)
BW_DRAM = 645e9    # bytes/s per socket (measured)
BW_L3 = 1.4e12     # bytes/s per socket aggregate (measured 3-4 MiB/core)
UPI = 1.0          # us per sync point / hop across the two sockets of one server (assumed)

def project(md, prec, sock, scen, net, ctx):
    bpp = 1 if prec == "int8" else 2
    cores = SOCKETS[sock]["cores"]
    sc, nt = SCEN[scen], NET[net]
    cap = cores * sc["l2"] * 1048576
    bw = cores * sc["bw"]
    lb = md["body"] * bpp / md["layers"]
    if lb <= cap:
        per = max(1, int(cap // lb))          # layers per socket
        T = 1
        stages = math.ceil(md["layers"] / per)
        sockets = stages
        gemv = lb / bw
        xs = 0.0
        hops = stages  # stage boundaries incl. return; alternate UPI / network on 2S servers
        hop_us = (stages // 2) * UPI + (stages - stages // 2) * nt["hop"]
    else:
        T = math.ceil(lb / cap)
        per = 1
        stages = md["layers"]
        sockets = stages * T
        gemv = lb / (T * bw)
        xs = UPI if T == 2 else nt["hop"]
        hop_us = stages * nt["hop"]
    sync = sc["points"] * (sc["sync"] + xs)
    layer_us = gemv * 1e6 + sync
    lmb = md["lm"] * bpp
    lmk = math.ceil(lmb / cap)
    lm_us = lmb / (lmk * bw) * 1e6 + 2 * sc["sync"] + 2 * nt["hop"]
    kvb = 1 if prec == "int8" else 2
    attn_g = md["g_elem"] * kvb * ctx / (T * BW_DRAM) * 1e6
    attn_s = md["s_elem"] * kvb * min(ctx, md["window"]) / (T * BW_L3) * 1e6 if md["slid"] else 0.0
    # KV spread: give each full-attention layer X >= T sockets so its KV read fits the stage time
    tgt = max(layer_us, 10.0)
    X = max(T, math.ceil(attn_g * T / tgt)) if attn_g > tgt else T
    attn_gx = attn_g * T / X + (nt["hop"] if X > T else 0.0)
    extra = md["glob"] * (X - T)
    X8 = max(T, 8) if attn_g > tgt else T
    attn_g8 = attn_g * T / X8 + (nt["hop"] if X8 > T else 0.0)
    attn_us = md["glob"] * attn_g + md["slid"] * attn_s
    attn_8 = md["glob"] * attn_g8 + md["slid"] * attn_s
    attn_x = md["glob"] * attn_gx + md["slid"] * attn_s
    total = md["layers"] * layer_us + hop_us + lm_us + attn_us
    total_x = md["layers"] * layer_us + hop_us + lm_us + attn_x
    socks = sockets + lmk
    return dict(model=md["name"], prec=prec, sock=sock, scen=scen, net=net, ctx=ctx, sockets=socks,
                servers=math.ceil(socks / 2), layer_bytes_mb=round(lb / 1e6, 1), layers_per_socket=per,
                sockets_per_layer=T, layer_us=round(layer_us, 1),
                budget_us=dict(gemv=round(md["layers"] * gemv * 1e6), sync=round(md["layers"] * sync),
                               hops=round(hop_us), lm_head=round(lm_us), attention=round(attn_us)),
                token_us=round(total), tok_s=round(1e6 / total),
                kv8=dict(sockets=socks + md["glob"] * (X8 - T), tok_s=round(1e6 / (md["layers"] * layer_us + hop_us + lm_us + attn_8))),
                kv_spread=dict(sockets=socks + extra, servers=math.ceil((socks + extra) / 2), per_global_layer=X,
                               token_us=round(total_x), tok_s=round(1e6 / total_x)))

def main():
    mods = [gemma(m) for m in BODY] + [LLAMA8]
    out = []
    for md in mods:
        for prec in ("int8", "bf16"):
            for sock in SOCKETS:
                for scen in SCEN:
                    for net in NET:
                        for ctx in (2048, 16384, 32768, 131072, 262144):
                            if md["name"].startswith("gemma-4-E") and ctx > 131072:
                                continue
                            out.append(project(md, prec, sock, scen, net, ctx))
    json.dump(dict(rows=out, sockets=SOCKETS, scen=SCEN, net=NET), open("/tmp/g4c/l2bw/project.json", "w"), indent=1)
    print(f"{'model':32} {'prec':4} {'sock':6} {'scen':5} {'net':7} {'ctx':>6} {'skt':>4} {'srv':>4} {'T':>2} {'per':>3} {'lay_us':>6} {'tok_us':>6} {'tok/s':>6}")
    for r in out:
        if r["ctx"] in (32768, 131072, 262144) and r["sock"] == "gnr128" and r["scen"] == "tuned":
            print(f"{r['model'][:32]:32} {r['prec']:4} {r['sock']:6} {r['scen']:5} {r['net']:7} {r['ctx']:6} {r['sockets']:4} {r['servers']:4} "
                  f"{r['sockets_per_layer']:2} {r['layers_per_socket']:3} {r['layer_us']:6} {r['token_us']:6} {r['tok_s']:6}  kv8 {r['kv8']['sockets']:4} {r['kv8']['tok_s']:5}  kvx {r['kv_spread']['sockets']:5} {r['kv_spread']['tok_s']:6}")

main()
