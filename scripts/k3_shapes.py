#!/usr/bin/env python3
"""Kimi-K3's prefill dense-GEMM shapes at TP8, computed from the config, in both encodings.

WHY THIS EXISTS RATHER THAN `--shapes auto`
-------------------------------------------
`plowc tune gemm --shapes auto` derives demand through the schedule-model path. K3's production
path is the hybrid devblob emitter, while the schedule path remains an analysis/refusal path, so
`auto` still cannot observe K3 demand. This generator is cross-checked against the emitted TP8
`model.pkt`: every unique `Gemm*` `(N,K)` in prefill must appear below.

SO WHAT IS THIS INSTEAD
-----------------------
Every N and K below is computed from named `config.json` fields and mirrors the production K3
devblob emitter. Change the config and the list moves with it. The checked-in output is also
validated against a real packet census so emitter fusion/splitting drift is visible.

TP8 SHARDING, as the task's contract states it and as `nn-graph` computes it:
  * the HEAD axis shards by 8 -- MLA's `nh`, KDA's `num_heads`, and the widths derived from them
    (`nh * qk_head`, `nh * (qk_nope + v_head)`, `nh * v_head`, `num_heads * head_dim`). When the
    head axis is the CONTRACTION dim (`o_proj`), it is K that shards, not N.
  * the EXPERT INTERMEDIATE shards by 8 -- `moe_intermediate_size` and the dense
    `intermediate_size`, in both the column (gate/up) and row (down) orientation.
  * the LATENT projections do NOT shard -- `q_lora_rank`, `kv_lora_rank`,
    `routed_expert_hidden_size`, and the router, are replicated.

BOTH ENCODINGS AT EVERY SHAPE. The campaign measures `None` and `Mxfp4` over the same (M,N,K), so
"is mxfp4 faster than bf16 here" is a question the store can answer per shape rather than per
model. It is not a rhetorical question -- the measured answer is no at some shapes.

M LADDER: the shipping prefill buckets (512, 1024, 2048, 4096, 8192; `MAX_CHUNK = 8192` in
`plowrt/src/exec/amd.rs` filters anything above), plus 128. 128 is not a serving bucket; it is
kept because it is the shape `op_gemm.h`'s own MXFP4 note is about -- "Kimi's mxfp4 kv_a_proj
(M=128, N=576) ran at ~0.4% of peak: at that shape 256x256 is THREE tiles on 256 CUs" -- and it
is the regime where tile QUANTISATION, not arithmetic intensity, decides.

    usage: k3_shapes.py [config.json | dir]     (default /tmp/k3-synth)
"""
import json
import os
import sys

TP = 8
M_LADDER = [128, 512, 1024, 2048, 4096, 8192]
QUANTS = ["None", "Mxfp4"]


def shapes(t):
    """Every (N, K, label) a K3 prefill layer asks `pick_tile` about, at TP8."""
    h = t["hidden_size"]
    nh = t["num_attention_heads"] // TP
    v_head = t["v_head_dim"]
    q_lora, kv_lora = t["q_lora_rank"], t["kv_lora_rank"]
    lac = t["linear_attn_config"]
    kda_inner = lac["num_heads"] * lac["head_dim"] // TP
    gate_rank = lac["head_dim"]
    latent = t["routed_expert_hidden_size"]
    shared = t["num_shared_experts"] * t["moe_intermediate_size"] // TP
    dense = t["intermediate_size"] // TP
    out = [
        # --- MLA, 24 layers. `mla()` in nn-graph/src/models/kimi_k3.rs.
        (q_lora, h, "mla-q-a"),
        (kv_lora, h, "mla-kv-a"),
        (t["qk_rope_head_dim"], h, "mla-k-rope-down"),
        (nh * kv_lora, q_lora, "mla-q-absorb"),
        (nh * t["qk_rope_head_dim"], q_lora, "mla-q-rope"),
        (nh * v_head, h, "mla-o-gate"),
        (h, nh * v_head, "mla-o"),
        # --- KDA, 69 layers. `kda()`.
        (kda_inner, h, "kda-qkvg"),                          # q|k|v|g_proj, same shape
        (gate_rank, h, "kda-f-a"),                           # forget-gate rank: not sharded
        (kda_inner, gate_rank, "kda-f-b"),
        (lac["num_heads"] // TP, h, "kda-beta"),             # one scalar per head
        (h, kda_inner, "kda-o"),                             # head axis on K
        # --- Latent MoE, 92 layers. `latent_moe()`.
        (t["num_experts"], h, "moe-router"),                 # replicated
        (latent, h, "moe-latent-down"),                      # latent: not sharded
        (h // TP, latent, "moe-latent-up"),
        (shared, h, "moe-shared-gateup"),
        (h, shared, "moe-shared-down"),
        # --- Dense FFN, first_k_dense_replace layers. `situ_mlp()`.
        (dense, h, "dense-gateup"),
        (h, dense, "dense-down"),
    ]
    # The per-expert projections are NOT here on purpose: routed experts lower to the grouped
    # ops (MoeGroupGluPf/MoeGroupDownPf, 85/86), which carry their own tile and never consult
    # `pick_tile`. Measuring them here would publish records nothing reads.
    seen, uniq = set(), []
    for n, k, label in out:
        if (n, k) in seen:
            continue
        seen.add((n, k))
        uniq.append((n, k, label))
    return uniq


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "/tmp/k3-synth"
    path = arg if arg.endswith(".json") else os.path.join(arg, "config.json")
    t = json.load(open(path))["text_config"]
    rows = shapes(t)
    print(f"# Kimi-K3 prefill dense-GEMM demand at TP{TP}, generated by scripts/k3_shapes.py")
    print(f"# from {path}. DO NOT HAND-EDIT: regenerate. See this script's header for why")
    print("# `--shapes auto` cannot produce it yet and what to do when it can.")
    print(f"# {len(rows)} distinct (N,K) x {len(M_LADDER)} M x {len(QUANTS)} encodings"
          f" = {len(rows) * len(M_LADDER) * len(QUANTS)} sweeps")
    for q in QUANTS:
        for n, k, label in rows:
            for m in M_LADDER:
                if q == "Mxfp4" and k % 64:
                    # The w4a16 B-fetch is KEXACT with a 32-element MX block; the harness
                    # refuses such a K rather than measure it wrong. K3 has none.
                    continue
                print(f"{m} {n} {k}    k3-{label}-M{m}    {q}")


if __name__ == "__main__":
    main()
