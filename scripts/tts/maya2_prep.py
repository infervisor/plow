#!/usr/bin/env python3
"""Maya 2 Native (maya-research/maya2-native, `maya3` package) -> a plow-compilable checkpoint directory.

  HF_HOME=... python scripts/tts/maya2_prep.py OUT_DIR REVISION

The backbone is Qwen3-1.7B (HF half-split RoPE, no permutation), so the directory is a Qwen3 checkpoint:
  model.*                        <- backbone.lm.model.* ; embed_tokens row <|audio|> (194499) carries the
                                    named/plain speaker anchor, bf16(embed + fc2(gelu(fc1(null_vec)))),
                                    which serving adds at every <|audio|> sentinel (a model constant)
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
"""
import argparse, glob, json, math, os, shutil

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

REPO = "models--maya-research--maya2-native"
AUDIO = 194499
CODES, LEVELS, GROUPS, LAYERS = 2048, 31, 11, 4
GROUP_OF_LEVEL = [None] + list(range(8)) + [8] * 8 + [9] * 8 + [10] * 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("revision")
    args = ap.parse_args()
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
    emb[AUDIO] = (emb[AUDIO].float() + a).to(torch.bfloat16)
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
    for l in range(LAYERS):
        v = F.linear(mem, f32(f"depth.cross.{l}.kv.weight"))[dim:]
        d[f"maya2.depth.cross.{l}"] = F.linear(v, f32(f"depth.cross.{l}.o.weight"))
    d["maya2.depth.norm.weight"] = f32("depth.norm.weight")
    d["maya2.depth.norm.bias"] = f32("depth.norm.bias")
    for j in range(LEVELS):
        d[f"maya2.depth.heads.{j}.weight"] = f32(f"depth.heads.{j}.weight")
    # Matrices fp16 (the engine's depth dtype); vectors stay fp32 (norms, cross constants, pos0).
    out.update({k: (v.to(torch.float16) if v.dim() == 2 else v).contiguous() for k, v in d.items()})
    save_file(out, f"{args.out}/model.safetensors")

    qwen = dict(
        architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=bb["hidden_size"],
        intermediate_size=bb["intermediate_size"], num_hidden_layers=bb["num_hidden_layers"],
        num_attention_heads=bb["num_attention_heads"], num_key_value_heads=bb["num_key_value_heads"],
        head_dim=bb["head_dim"], rms_norm_eps=bb["rms_norm_eps"], rope_theta=bb["rope_parameters"]["rope_theta"],
        rope_scaling=None, vocab_size=bb["vocab_size"], max_position_embeddings=bb["max_position_embeddings"],
        tie_word_embeddings=False, torch_dtype="bfloat16", bos_token_id=None, eos_token_id=None,
        plow_head_rows=CODES + 1,
        maya2=dict(
            audio_token=AUDIO, lang_tokens={l: 194504 + i for i, l in enumerate(
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


if __name__ == "__main__":
    main()
