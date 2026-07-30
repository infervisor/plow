#!/usr/bin/env python3
"""Kimi-K3's prefill dense-GEMM shapes at TP8, computed from the config, in both encodings.

WHY THIS EXISTS RATHER THAN `--shapes auto`
-------------------------------------------
`plowc tune gemm --shapes auto` derives a campaign's shape list by RUNNING a real emit and reading
back every `pick_tile` lookup, which is the only construction that cannot drift from the compiler.
It does not work for K3, and the reason is recorded in the tree rather than guessed:
`devgen::mla::kimi_k3_emit` is an analysis-and-refusal path, not an emitter. It prints

    MISSING CAPABILITIES - 2 of them, ranked (blocker first).
     1. full-model emit for a hybrid MLA arch - THE ONE REMAINING BLOCKER  [the whole blob]
        ... `crates/devgen/src/k3.rs` and `crates/devgen/src/kda.rs` are reached by NOTHING
        outside their own `#[cfg(test)]` modules ... No function composes them into even ONE
        complete layer, and there is no loop over layers anywhere.

and then panics. It never reaches a single `pick_tile`, so there is no demand log to read: `auto`
returns "the emit asked the tuning store about no dense GEMM at all", which is that command
correctly refusing to invent a campaign.

WHEN THAT BLOCKER CLOSES, DELETE THIS FILE and switch `scripts/rebench_tune_gemm_all.sh` to
`--shapes auto`. A generator is second best and this header should not outlive its reason.

SO WHAT IS THIS INSTEAD
-----------------------
The next best thing, and deliberately not a hand-typed list: every N and K below is COMPUTED from
named `config.json` fields, and the shapes are read off `crates/nn-graph/src/models/kimi_k3.rs`'s
graph builder -- the one place in the tree that does state K3's projections completely (`mla()`,
`kda()`, `situ_mlp()`, `latent_moe()`). Change the config and the list moves with it. What it
CANNOT catch is the emitter choosing to fuse, split or skip a projection, which is exactly the
class of drift `auto` exists to close.

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

M LADDER: the shipping prefill buckets (512, 2048, 4096, 8192; `MAX_CHUNK = 8192` in
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
M_LADDER = [128, 512, 2048, 4096, 8192]
QUANTS = ["None", "Mxfp4"]


def shapes(t):
    """Every (N, K, label) a K3 prefill layer asks `pick_tile` about, at TP8."""
    h = t["hidden_size"]
    nh = t["num_attention_heads"] // TP
    qk_head = t["qk_nope_head_dim"] + t["qk_rope_head_dim"]
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
        (q_lora, h, "mla-q-a"),                              # latent: not sharded
        (nh * qk_head, q_lora, "mla-q-b"),                   # head axis on N
        (kv_lora + t["qk_rope_head_dim"], h, "mla-kv-a"),    # latent: not sharded
        (nh * (t["qk_nope_head_dim"] + v_head), kv_lora, "mla-kv-b"),
        (nh * v_head, h, "mla-o-gate"),                      # K3's MLA output gate
        (h, nh * v_head, "mla-o"),                           # head axis on K
        # --- KDA, 69 layers. `kda()`.
        (kda_inner, h, "kda-qkvg"),                          # q|k|v|g_proj, same shape
        (gate_rank, h, "kda-f-a"),                           # forget-gate rank: not sharded
        (kda_inner, gate_rank, "kda-f-b"),
        (lac["num_heads"] // TP, h, "kda-beta"),             # one scalar per head
        (h, kda_inner, "kda-o"),                             # head axis on K
        # --- Latent MoE, 92 layers. `latent_moe()`.
        (t["num_experts"], h, "moe-router"),                 # replicated
        (latent, h, "moe-latent-down"),                      # latent: not sharded
        (h, latent, "moe-latent-up"),                        # latent: not sharded
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
