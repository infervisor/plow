#!/usr/bin/env python3
"""HF transformers reference for multimodal chat (images / audio) on a Gemma-4-style checkpoint.

  preprocess  CPU: processor outputs per item (pixel patches + positions, log-mel + valid frames)
  all         + GPU: projected soft tokens per item (get_image_features / get_audio_features) and
              greedy completions of chat cases (`--cases`: [{name, media: [item index], text}])

Writes `<out>/ref.json` plus raw little-endian f32 arrays (`<out>/<name>.f32`); `crates/plowrt/
examples/mm_check.rs` and `scripts/mm/gate.py` read them. WAV input is 16-bit PCM.
"""
import argparse, base64, json, os, sys, wave

import numpy as np


def save(out, name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(os.path.join(out, name + ".f32"))
    return {"file": name + ".f32", "shape": list(arr.shape)}


def read_wav(path):
    w = wave.open(path)
    assert w.getsampwidth() == 2 and w.getnchannels() == 1, path
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0, w.getframerate()


def preprocess(proc, items, out):
    from PIL import Image
    res = []
    for i, it in enumerate(items):
        if it["kind"] == "image":
            img = Image.open(it["path"]).convert("RGB")
            o = proc.image_processor(images=[img], return_tensors="np")
            pos = o["image_position_ids"][0]
            n = int((pos[:, 0] >= 0).sum())
            r = dict(it, patches=save(out, f"pix{i}", o["pixel_values"][0][:n]), positions=pos[:n].tolist(),
                     soft_tokens=int(o["num_soft_tokens_per_image"][0]), size=[img.height, img.width])
        else:
            pcm, rate = read_wav(it["path"])
            o = proc.feature_extractor([pcm], sampling_rate=rate, return_tensors="np")
            mask = o["input_features_mask"][0].astype(bool)
            valid = int(mask.sum())
            r = dict(it, mel=save(out, f"mel{i}", o["input_features"][0][:valid]), valid_frames=valid,
                     frames=int(mask.shape[0]))
        res.append(r)
    return res


def encode(model, proc, items, out):
    import torch
    from PIL import Image
    dev = model.device
    for i, it in enumerate(items):
        with torch.no_grad():
            if it["kind"] == "image":
                img = Image.open(it["path"]).convert("RGB")
                o = proc.image_processor(images=[img], return_tensors="pt")
                f = model.model.get_image_features(o["pixel_values"].to(dev, model.dtype), o["image_position_ids"].to(dev),
                                                   return_dict=True)
                rows = torch.cat(f.pooler_output, dim=0)
            else:
                pcm, rate = read_wav(it["path"])
                o = proc.feature_extractor([pcm], sampling_rate=rate, return_tensors="pt")
                f = model.model.get_audio_features(o["input_features"].to(dev, model.dtype),
                                                   o["input_features_mask"].to(dev), return_dict=True)
                rows = f.pooler_output[f.attention_mask.to(dev)]
        it["encoded"] = save(out, f"enc{i}", rows.float().cpu().numpy())


def media_part(it):
    b64 = base64.b64encode(open(it["path"], "rb").read()).decode()
    if it["kind"] == "image":
        mime = "image/png" if it["path"].endswith(".png") else "image/jpeg"
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}


def generate(model, proc, cases, items, max_new):
    import torch
    from PIL import Image
    out = []
    for c in cases:
        media = [items[k] for k in c["media"]]
        content = [{"type": it["kind"]} for it in media] + [{"type": "text", "text": c["text"]}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        images = [Image.open(it["path"]).convert("RGB") for it in media if it["kind"] == "image"]
        audio = [read_wav(it["path"])[0] for it in media if it["kind"] == "audio"]
        inputs = proc(text=[text], images=[images] if images else None, audio=audio or None,
                      return_tensors="pt").to(model.device)
        for k in ("pixel_values", "input_features"):
            if k in inputs:
                inputs[k] = inputs[k].to(model.dtype)
        with torch.no_grad():
            g = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        new = g[0][inputs["input_ids"].shape[1]:].tolist()
        out.append(dict(c, prompt_tokens=int(inputs["input_ids"].shape[1]), tokens=new,
                        output=proc.tokenizer.decode(new, skip_special_tokens=True)))
        print(c["name"], repr(out[-1]["output"][:120]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preprocess", "all"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--audio", action="append", default=[])
    ap.add_argument("--cases")
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(args.ckpt)
    items = [{"kind": "image", "path": os.path.abspath(p)} for p in args.image]
    items += [{"kind": "audio", "path": os.path.abspath(p)} for p in args.audio]
    ref = {"ckpt": args.ckpt, "items": preprocess(proc, items, args.out)}
    if args.stage == "all":
        import torch
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.ckpt, dtype=torch.bfloat16).to("cuda").eval()
        encode(model, proc, ref["items"], args.out)
        if args.cases:
            cases = json.load(open(args.cases))
            ref["cases"] = generate(model, proc, cases, ref["items"], args.max_new)
            for c in ref["cases"]:
                c["messages"] = [{"role": "user", "content": [media_part(ref["items"][k]) for k in c["media"]] +
                                  [{"type": "text", "text": c["text"]}]}]
    json.dump(ref, open(os.path.join(args.out, "ref.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
