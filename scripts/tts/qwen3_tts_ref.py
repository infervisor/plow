#!/usr/bin/env python3
"""Qwen3-TTS (12Hz, CustomVoice) reference: the official `qwen_tts` model driven by an explicit
streaming-text loop (`non_streaming_mode=False`, the tts-bench request), dumping the per-frame codes,
the talker / code-predictor logits and the decoded PCM that the plow packet is gated against.

  python scripts/tts/qwen3_tts_ref.py MODEL_DIR TEXTS.jsonl OUT_DIR [--voice ryan] [--language english]
      [--dtype bf16|fp32] [--device cuda] [--seed S | greedy] [--logits] [--official] [--limit N]
      [--force CODES_DIR]

TEXTS.jsonl rows: {"id": ..., "text": ...}. Per row OUT_DIR gets `<id>.codes.npy` (int32 [frames][16],
EOS frame excluded), `<id>.wav` (24 kHz), with --logits `<id>.logits.npz` (cb0 [frames+1][3072] after
the reference's suppression, cp [frames][15][2048], prefill [rows][2048] input embeddings), and one
line of OUT_DIR/meta.jsonl (frames, audible onset ms, seconds, wall ms). --official also runs the
package's own `generate` with the same settings and records whether its codes equal the loop's.
--force teacher-forces every frame to CODES_DIR/<id>.codes.npy (then EOS), so logits of two runs
(dtypes, or plow) compare position by position; the logits then also hold `cb0_raw`, the head's
3072 logits before suppression and penalty.
--prepared-predictor --dtype fp32 takes a prepared checkpoint and packet predictor capture JSON;
it writes conditional FP32 logits and comparison diagnostics without loading the audio model.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

EOS, CODES, VOCAB = 2150, 2048, 3072


def load_prepared_predictor(path, codebook, device):
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3Model
    from transformers.modeling_utils import AttentionInterface
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm"))
    from fp32_ref_gate import fp32_attention

    AttentionInterface.register("sdpa", fp32_attention)
    path = Path(path)
    meta = json.loads((path / "config.json").read_text())["plow_frame_embed"]
    d = meta["depth"]
    assert d["qk_norm"] and 0 < codebook < meta["codebooks"]
    cfg = Qwen3Config(vocab_size=1, hidden_size=d["hidden"], intermediate_size=d["intermediate"],
                      num_hidden_layers=d["layers"], num_attention_heads=d["heads"],
                      num_key_value_heads=d["kv_heads"], head_dim=d["head_dim"],
                      rms_norm_eps=d["eps"], rope_theta=d["rope_theta"],
                      rope_scaling=d["rope_scaling"], max_position_embeddings=d["positions"],
                      tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    model = Qwen3Model(cfg).float().eval().to(device)
    prefix = meta["prefix"]
    weights = {}
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as f:
        state = {name: (torch.zeros_like(value) if name == "embed_tokens.weight" else
                        f.get_tensor(f"{prefix}.{name}").float().to(device))
                 for name, value in model.state_dict().items()}
        model.load_state_dict(state, strict=True)
        for name, key in (("projection", f"{prefix}.proj.weight"),
                          ("head", f"{prefix}.head.{codebook - 1}.weight"),
                          ("embeddings", meta["table"])):
            weights[name] = f.get_tensor(key).float().to(device)
        weights["bias"] = f.get_tensor(f"{prefix}.proj.bias").float().to(device) if d["proj_bias"] else None
    assert weights["embeddings"].shape == (meta["codebooks"] * meta["codebook_size"],
                                             weights["projection"].shape[1])
    assert weights["head"].shape == (meta["classes"], d["hidden"])
    return model, weights, meta


@torch.inference_mode()
def prepared_predictor_logits(model, weights, meta, case, codebook):
    device = weights["projection"].device
    codes = case["prefix_codes"]
    assert len(codes) == codebook and all(0 <= c < meta["classes"] for c in codes)
    words = np.asarray(case["hidden_bf16"], dtype=np.uint16).astype(np.uint32) << 16
    hidden = torch.from_numpy(words.view(np.float32).copy()).to(device)
    assert hidden.shape == (weights["projection"].shape[1],) and hidden.isfinite().all()
    ids = torch.tensor([j * meta["codebook_size"] + c for j, c in enumerate(codes)], device=device)
    x = torch.cat((hidden[None], weights["embeddings"][ids]), dim=0)[None]
    x = torch.nn.functional.linear(x, weights["projection"], weights["bias"])
    out = model(inputs_embeds=x, use_cache=False).last_hidden_state[:, -1]
    return torch.nn.functional.linear(out, weights["head"])[0]


def prepared_predictor_reference(a):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm"))
    from fp32_ref_gate import file_sha, kl_top

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    data = json.loads(Path(a.texts).read_text())
    assert data["version"] == 1 and data["prefix_selection_matches_full"]
    codebook = data["codebook"]
    model, weights, meta = load_prepared_predictor(a.model, codebook, a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for index, case in enumerate(data["cases"]):
        logits = prepared_predictor_logits(model, weights, meta, case, codebook)
        repeat = prepared_predictor_logits(model, weights, meta, case, codebook)
        assert logits.isfinite().all() and torch.equal(logits, repeat)
        actual = torch.tensor(case["logits"], dtype=torch.float32, device=a.device)
        assert actual.shape == logits.shape and actual.isfinite().all()
        def logprobs(x):
            return list(enumerate(torch.log_softmax(x.double(), -1).cpu().tolist()))
        delta = actual.double() - logits.double()
        np.savez_compressed(out / f"case-{index}.npz", reference=logits.cpu().numpy(),
                            packet=actual.cpu().numpy())
        top = torch.topk(logits, 5)
        records.append(dict(id=case["id"], selected=case["selected"],
                            kl=kl_top(logprobs(logits), logprobs(actual)),
                            max_abs=delta.abs().max().item(),
                            rel_l2=(delta.norm() / logits.double().norm().clamp_min(1e-30)).item(),
                            reference_top=list(zip(top.indices.cpu().tolist(), top.values.cpu().tolist())),
                            repeat_exact=True))
    report = dict(version=1, scope="conditional FP32 predictor at captured packet hidden/prefix inputs",
                  full_model_qualified=False, frame=data["frame"], codebook=codebook, cases=records,
                  hashes={str(p): file_sha(p) for p in [Path(a.texts), Path(a.model) / "config.json",
                          Path(a.model) / "model.safetensors", Path(__file__),
                          Path(__file__).resolve().parents[1] / "llm/fp32_ref_gate.py"]},
                  torch=torch.__version__, device=a.device)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records), flush=True)


def onset_ms(pcm):
    """tts-bench AudibleOnsetDetector v1: 20 ms frames, 10 ms hop, DC-removed RMS >= -45 dBFS twice."""
    x = np.clip(np.round(pcm * 32767.0), -32768, 32767).astype(np.int16).astype(np.float32) / 32768.0
    f, h, thr = 480, 240, 10 ** (-45 / 20)
    run, start = 0, None
    for s in range(0, len(x) - f + 1, h):
        w = x[s:s + f] - x[s:s + f].mean()
        if np.sqrt(np.mean(w * w)) >= thr:
            if run == 0:
                start = s
            run += 1
            if run >= 2:
                return start * 1000.0 / 24000
        else:
            run = 0
    return None


class Loop:
    def __init__(self, m, tok, voice, language):
        self.m, self.t, self.cfg = m, m.talker, m.config
        tc = self.cfg.talker_config
        self.tok = tok
        self.dev = next(m.parameters()).device
        self.spk = tc.spk_id[voice.lower()]
        lang = tc.codec_language_id[language.lower()]
        self.codec_prompt = [tc.codec_think_id, tc.codec_think_bos_id, lang, tc.codec_think_eos_id,
                             self.spk, tc.codec_pad_id, tc.codec_bos_id]

    def T(self, ids):
        ids = torch.tensor([ids], device=self.dev)
        return self.t.text_projection(self.t.get_text_embeddings()(ids))

    def C(self, ids):
        return self.t.get_input_embeddings()(torch.tensor([ids], device=self.dev))

    def prompt(self, text):
        ids = self.tok(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")["input_ids"]
        bos, eos, pad = self.T([self.cfg.tts_bos_token_id, self.cfg.tts_eos_token_id,
                                self.cfg.tts_pad_token_id]).chunk(3, dim=1)
        codec = self.C(self.codec_prompt)
        n = codec.shape[1]
        head = torch.cat([pad.expand(-1, n - 2, -1), bos], dim=1) + codec[:, :-1]
        rows = torch.cat([self.T(ids[:3]), head, self.T(ids[3:4]) + codec[:, -1:]], dim=1)
        trailing = torch.cat([self.T(ids[4:-5]), eos], dim=1)
        return ids, rows, trailing, pad

    def sample(self, logits, greedy, gen, top_k=50, temperature=0.9):
        if greedy:
            return int(torch.argmax(logits).item())
        v, i = torch.topk(logits.float() / temperature, top_k)
        p = torch.softmax(v, -1)
        return int(i[torch.multinomial(p, 1, generator=gen)].item())

    @torch.inference_mode()
    def run(self, text, greedy, seed, max_frames, dump, force=None):
        from transformers import DynamicCache
        gen = torch.Generator(device=self.dev)
        gen.manual_seed(seed)
        ids, rows, trailing, pad = self.prompt(text)
        cache = DynamicCache()
        L = rows.shape[1]
        pos = torch.arange(L, device=self.dev).view(1, 1, -1).expand(3, 1, -1)
        out = self.t.model(inputs_embeds=rows, position_ids=pos, past_key_values=cache, use_cache=True)
        h = out.last_hidden_state[:, -1:]
        cp = self.t.code_predictor
        frames, l0s, lcps, hist, raws = [], [], [], [], []
        for step in range(max_frames if force is None else len(force) + 1):
            logits = self.t.codec_head(h)[0, -1].float()
            raws.append(logits.clone().cpu())
            logits[CODES:] = -float("inf")
            if step >= 2:
                logits[EOS] = self.t.codec_head(h)[0, -1, EOS].float()
            for t_ in set(hist):
                logits[t_] = logits[t_] * 1.05 if logits[t_] < 0 else logits[t_] / 1.05
            l0s.append(logits.cpu())
            c0 = self.sample(logits, greedy, gen)
            if force is not None:
                c0 = int(force[step][0]) if step < len(force) else EOS
            if c0 == EOS:
                break
            hist.append(c0)
            codes, lcp = [c0], []
            e0 = self.t.get_input_embeddings()(torch.tensor([[c0]], device=self.dev))
            cpc = DynamicCache()
            x = cp.small_to_mtp_projection(torch.cat([h, e0], dim=1))
            for j in range(15):
                o = cp.model(inputs_embeds=x, past_key_values=cpc, use_cache=True)
                lg = cp.lm_head[j](o.last_hidden_state[:, -1]).float()[0]
                lcp.append(lg.cpu())
                c = self.sample(lg, greedy, gen)
                if force is not None:
                    c = int(force[step][j + 1])
                codes.append(c)
                x = cp.small_to_mtp_projection(cp.model.get_input_embeddings()[j](
                    torch.tensor([[c]], device=self.dev)))
            frames.append(codes)
            lcps.append(torch.stack(lcp))
            # The reference sums the 16 rows in one reduction (`codec_hiddens.sum(1)`).
            emb = torch.cat([e0] + [cp.model.get_input_embeddings()[j](
                torch.tensor([[codes[j + 1]]], device=self.dev)) for j in range(15)], dim=1).sum(1, keepdim=True)
            emb = emb + (trailing[:, step:step + 1] if step < trailing.shape[1] else pad)
            p = torch.full((3, 1, 1), L + step, device=self.dev)
            h = self.t.model(inputs_embeds=emb, position_ids=p, past_key_values=cache,
                             use_cache=True).last_hidden_state
        res = dict(ids=ids, codes=np.array(frames, dtype=np.int32).reshape(-1, 16))
        if dump:
            res["logits"] = dict(cb0=torch.stack(l0s).numpy(), cb0_raw=torch.stack(raws).numpy(),
                                 cp=(torch.stack(lcps).numpy() if lcps else np.zeros((0, 15, CODES), np.float32)),
                                 prefill=rows[0].float().cpu().numpy())
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("texts")
    ap.add_argument("out")
    ap.add_argument("--voice", default="ryan")
    ap.add_argument("--language", default="english")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=None, help="sample with this seed (default greedy)")
    ap.add_argument("--max-frames", type=int, default=2048)
    ap.add_argument("--logits", action="store_true")
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", default=None)
    ap.add_argument("--prepared-predictor", action="store_true",
                    help="FP32 conditional predictor: MODEL=prepared checkpoint, TEXTS=packet predictor capture")
    a = ap.parse_args()
    if a.prepared_predictor:
        if a.dtype != "fp32":
            ap.error("--prepared-predictor requires --dtype fp32")
        prepared_predictor_reference(a)
        return
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    w = Qwen3TTSModel.from_pretrained(a.model, dtype=dt, device_map=a.device)
    tok = w.processor.tokenizer if hasattr(w.processor, "tokenizer") else w.processor
    loop = Loop(w.model, tok, a.voice, a.language)
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.texts) if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    greedy = a.seed is None
    meta = open(os.path.join(a.out, "meta.jsonl"), "a")
    for r in rows:
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        force = np.load(os.path.join(a.force, f"{r['id']}.codes.npy")) if a.force else None
        res = loop.run(r["text"], greedy, a.seed or 0, a.max_frames, a.logits, force)
        codes = res["codes"]
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        wavs, sr = w.model.speech_tokenizer.decode([{"audio_codes": torch.from_numpy(codes).long().to(a.device)}])
        pcm = np.asarray(wavs[0], dtype=np.float32)
        sf.write(os.path.join(a.out, f"{r['id']}.wav"), pcm, sr)
        np.save(os.path.join(a.out, f"{r['id']}.codes.npy"), codes)
        if a.logits:
            np.savez_compressed(os.path.join(a.out, f"{r['id']}.logits.npz"), **res["logits"])
        rec = dict(id=r["id"], text=r["text"], ids=res["ids"], frames=int(codes.shape[0]),
                   seconds=len(pcm) / sr, onset_ms=onset_ms(pcm), wall_ms=round(ms, 1),
                   dtype=a.dtype, seed=a.seed, voice=a.voice)
        if a.official:
            if not greedy:
                torch.manual_seed(a.seed)
            off, _ = w.model.generate(
                input_ids=[torch.tensor([res["ids"]], device=a.device)], instruct_ids=[None],
                languages=[a.language], speakers=[a.voice], non_streaming_mode=False,
                do_sample=not greedy, subtalker_dosample=not greedy, top_k=50, top_p=1.0,
                temperature=0.9, subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9,
                repetition_penalty=1.05, max_new_tokens=a.max_frames)
            oc = off[0].cpu().numpy().astype(np.int32)
            n = min(len(oc), len(codes))
            rec["official_frames"] = int(len(oc))
            rec["official_prefix_match"] = int(next((i for i in range(n) if not (oc[i] == codes[i]).all()), n))
        meta.write(json.dumps(rec) + "\n")
        meta.flush()
        print(json.dumps({k: rec[k] for k in rec if k != "ids"}), flush=True)


if __name__ == "__main__":
    main()
