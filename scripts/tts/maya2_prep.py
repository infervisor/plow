#!/usr/bin/env python3
"""Maya 2 Native (maya-research/maya2-native, `maya3` package) -> a plow-compilable checkpoint directory.

  HF_HOME=... python scripts/tts/maya2_prep.py OUT_DIR REVISION --maya3-src MAYA3_SRC [--controller-only]

MAYA3_SRC is a checkout of the `maya3` package (Apache-2.0); its frame and duration priors become
`maya2_controller.json` (the serving controller's constants, `crates/devgen/src/tts.rs` MAYA2).

The backbone is Qwen3-1.7B (HF half-split RoPE, no permutation), so the directory is a Qwen3 checkpoint:
  model.*                        <- backbone.lm.model.* ; embed_tokens row <|audio|> (194499) carries the
                                    named/plain speaker anchor, bf16(embed + fc2(gelu(fc1(null_vec)))),
                                    which serving adds at every <|audio|> sentinel (a model constant);
                                    row 194548 (past the tokenizer) is the AEOS input row RMSNorm(aeos_in),
                                    and 2048 zero rows from 194549 are the ids of prompt frame rows
                                    (frame row of class c = 194549 + c; their codes embed, not the row)
  lm_head.weight                 <- backbone.cb0_head.weight [2049][2048]: classes 0..2047 = codebook 0,
                                    2048 = AEOS (stop); config `plow_head_rows` = 2049
  maya2.codebook_embed.weight    <- audio_in.table.weight [32*2048][2048] (frame feedback: sum / sqrt(32),
                                    then RMSNorm eps FLT_EPSILON with maya2.frame_norm.weight)
  maya2.frame_norm.weight        <- audio_in.norm.weight (fp32)
Depth (DepthV2, matrices fp16 as the official engine serves it, vectors fp32; folds in fp32):
  maya2.depth.in_proj.weight     [1024][2048]       backbone hidden -> depth width
  maya2.depth.pos0               [1024]             level_emb[0]
  maya2.depth.emb.weight         [31*2048][1024]    code_up(code_lo[k*2048 + c]) + level_emb[k + 1]
  maya2.depth.groups.{g}.{l}.{n1,n2}.{weight,bias} / .qkv.weight / .o.weight / .f1.weight / .f2.weight
  maya2.depth.cross.{l}          [1024]             o_l(v_l(null_mem)): the one-token cross-attention
                                                    (named/plain voices), added after layer l's FFN
  maya2.depth.norm.{weight,bias} / maya2.depth.heads.{0..30}.weight [2048][1024]
Host-side speaker tables (fp32, OUT_DIR/speaker/model.safetensors, read by plowrt for cloned voices):
  anchor.fc{1,2}.{weight,bias}   the speaker anchor MLP (192 -> 512 -> 2048, GELU erf)
  audio_row                      embed_tokens[<|audio|>] before the null anchor is baked in
  memory.{l}                     [1024][192] o_l @ v_l @ spk_proj_ecapa: the depth memory constant of an
                                 ECAPA embedding e is memory.{l} @ e (f16-rounded, as the engine's C)
"""
import argparse, glob, json, math, os, shutil

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

REPO = "models--maya-research--maya2-native"
AUDIO = 194499
AEOS_ROW = 194548
FRAME_LO = AEOS_ROW + 1
CODES, LEVELS, GROUPS, LAYERS = 2048, 31, 11, 4
GROUP_OF_LEVEL = [None] + list(range(8)) + [8] * 8 + [9] * 8 + [10] * 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("revision")
    ap.add_argument("--maya3-src", required=True)
    ap.add_argument("--controller-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    write_controller(args.out, args.maya3_src)
    if args.controller_only:
        return
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    src = os.path.join(hub, REPO, "snapshots", args.revision)
    mc = json.load(open(f"{src}/maya3_config.json"))
    bb = json.load(open(f"{src}/backbone/config.json"))
    assert mc["audio"]["n_codebooks"] == 32 and mc["audio"]["codebook_size"] == CODES
    sd = {}
    for f in sorted(glob.glob(f"{src}/model-*.safetensors")):
        sd.update(load_file(f))
    pre = mc["key_prefix"]
    sd = {k[len(pre):] if k.startswith(pre) else k: v for k, v in sd.items()}
    os.makedirs(args.out, exist_ok=True)
    f32 = lambda k: sd[k].float()

    out = {k[len("backbone.lm."):]: v for k, v in sd.items() if k.startswith("backbone.lm.model.")}
    # Named / plain anchor: MLP(null_vec) in fp32 (the reference runs it on bf16(null_vec)).
    x = sd["speaker_anchor.null_vec"].to(torch.bfloat16).float()
    a = F.linear(F.gelu(F.linear(x, f32("speaker_anchor.fc1.weight"), f32("speaker_anchor.fc1.bias"))),
                 f32("speaker_anchor.fc2.weight"), f32("speaker_anchor.fc2.bias"))
    emb = out["model.embed_tokens.weight"].clone()
    assert emb.shape[0] == AEOS_ROW
    spk = {
        "anchor.fc1.weight": f32("speaker_anchor.fc1.weight"), "anchor.fc1.bias": f32("speaker_anchor.fc1.bias"),
        "anchor.fc2.weight": f32("speaker_anchor.fc2.weight"), "anchor.fc2.bias": f32("speaker_anchor.fc2.bias"),
        "audio_row": emb[AUDIO].float(),
    }
    emb[AUDIO] = (emb[AUDIO].float() + a).to(torch.bfloat16)
    # nn.RMSNorm(eps=None): fp32, eps FLT_EPSILON, fp32 weight multiply, one rounding.
    x = f32("audio_in.aeos_in").reshape(-1)
    aeos = x * torch.rsqrt(x.pow(2).mean() + torch.finfo(torch.float32).eps) * f32("audio_in.norm.weight")
    emb = torch.cat([emb, aeos.to(torch.bfloat16)[None], torch.zeros(CODES, emb.shape[1], dtype=emb.dtype)])
    out["model.embed_tokens.weight"] = emb
    out["lm_head.weight"] = sd["backbone.cb0_head.weight"]
    out["maya2.codebook_embed.weight"] = sd["audio_in.table.weight"]
    out = {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}
    out["maya2.frame_norm.weight"] = f32("audio_in.norm.weight").contiguous()

    d = {}
    d["maya2.depth.in_proj.weight"] = f32("depth.in_proj.weight")
    lev = f32("depth.level_emb.weight")
    d["maya2.depth.pos0"] = lev[0]
    lo = f32("depth.code_lo.weight").view(LEVELS, CODES, -1)
    d["maya2.depth.emb.weight"] = (lo @ f32("depth.code_up.weight").T + lev[1:LEVELS + 1, None]).reshape(
        LEVELS * CODES, -1)
    for g in range(GROUPS):
        for l in range(LAYERS):
            p = f"depth.groups.{g}.{l}"
            for s in ("n1.weight", "n1.bias", "n2.weight", "n2.bias", "qkv.weight", "o.weight", "f1.weight",
                      "f2.weight"):
                d[f"maya2.{p}.{s}"] = f32(f"{p}.{s}")
    mem = f32("depth.null_mem").reshape(-1)
    dim = mem.shape[0]
    proj = f32("depth.spk_proj_ecapa.weight")
    for l in range(LAYERS):
        kv, o = f32(f"depth.cross.{l}.kv.weight"), f32(f"depth.cross.{l}.o.weight")
        v = F.linear(mem, kv)[dim:]
        d[f"maya2.depth.cross.{l}"] = F.linear(v, o)
        spk[f"memory.{l}"] = (o.double() @ kv[dim:].double() @ proj.double()).float()
    d["maya2.depth.norm.weight"] = f32("depth.norm.weight")
    d["maya2.depth.norm.bias"] = f32("depth.norm.bias")
    for j in range(LEVELS):
        d[f"maya2.depth.heads.{j}.weight"] = f32(f"depth.heads.{j}.weight")
    # Matrices fp16 (the engine's depth dtype); vectors stay fp32 (norms, cross constants, pos0).
    out.update({k: (v.to(torch.float16) if v.dim() == 2 else v).contiguous() for k, v in d.items()})
    save_file(out, f"{args.out}/model.safetensors")
    os.makedirs(f"{args.out}/speaker", exist_ok=True)
    save_file({k: v.contiguous() for k, v in spk.items()}, f"{args.out}/speaker/model.safetensors")

    qwen = dict(
        architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=bb["hidden_size"],
        intermediate_size=bb["intermediate_size"], num_hidden_layers=bb["num_hidden_layers"],
        num_attention_heads=bb["num_attention_heads"], num_key_value_heads=bb["num_key_value_heads"],
        head_dim=bb["head_dim"], rms_norm_eps=bb["rms_norm_eps"], rope_theta=bb["rope_parameters"]["rope_theta"],
        rope_scaling=None, vocab_size=FRAME_LO + CODES, max_position_embeddings=bb["max_position_embeddings"],
        tie_word_embeddings=False, torch_dtype="bfloat16", bos_token_id=None, eos_token_id=None,
        plow_head_rows=CODES + 1,
        maya2=dict(
            audio_token=AUDIO, aeos_token=AEOS_ROW, frame_token_lo=FRAME_LO, overlay_rows=2, lang_tokens={l: 194504 + i for i, l in enumerate(
                ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"])},
            stop_class=CODES, codebooks=32, codebook_size=CODES, frame_scale=1.0 / math.sqrt(32),
            frame_norm_eps=float(torch.finfo(torch.float32).eps), sample_rate=24000, frame_rate=12.5,
            depth=dict(layers=LAYERS, hidden=dim, heads=16, head_dim=64, intermediate=3072, eps=1e-5,
                       levels=LEVELS, groups=GROUPS, group_of_level=GROUP_OF_LEVEL[1:]),
            sampling=dict(cb0_temperature=0.9, cb0_top_k=64, presence_penalty=0.8, penalty_window=12,
                          stop_p=0.35, stop_hazard=0.6, depth_temperature=0.7, depth_top_k=64),
        ),
    )
    json.dump(qwen, open(f"{args.out}/config.json", "w"), indent=1)
    json.dump({}, open(f"{args.out}/generation_config.json", "w"))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(f"{src}/backbone/{f}", f"{args.out}/{f}")
    shutil.copy(f"{src}/voices/voices.json", f"{args.out}/voices.json")
    print(f"wrote {args.out}: {len(out)} tensors")


def write_controller(out, maya3_src):
    """Constants of `maya3.engine` (EngineConfig, `_cb0`, `_plan`) and `maya3.guards` (GuardConfig, DurationPrior)."""
    fp = json.load(open(f"{maya3_src}/configs/text_frames_prior.json"))
    dp = json.load(open(f"{maya3_src}/maya3/assets/duration_prior.json"))
    ctl = dict(
        cb0=dict(temperature=0.9, top_k=64, top_p=1.0, presence_penalty=0.8, penalty_window=12, stop_p=0.35,
                 stop_hazard=0.6, min_frames_floor=4),
        guard=dict(dur_k=2.2, dur_margin_s=1.2, stall=32, period_win=64, period_thr=0.7, pmin=2, pmax=16, start=32,
                   every=2, boost=4.0, backstop=24, hist=96, hard_cap=1500),
        frame_s=0.08, chunk_s=20.0, gap_s=0.15,
        frames_prior={l: [v["affine_a"], v["affine_b"], v["resid_p99"]] for l, v in fp.items()},
        duration_prior={l: v["coef"] for l, v in dp["langs"].items()},
    )
    json.dump(ctl, open(f"{out}/maya2_controller.json", "w"), indent=1)


if __name__ == "__main__":
    main()
