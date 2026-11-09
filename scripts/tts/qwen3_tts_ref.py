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
--decoder-reference --dtype fp32 takes a tokenizer snapshot and saved encoder-reference JSON;
--decoder-result PATH (repeatable) compares captured priming PCM against the official decoder.
--mimi-streaming-source PATH uses the pinned upstream StaticMimi source for the streaming
reference; the raw HF eager decoder otherwise uses full-history causal attention.
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


def load_reference_decoder(path, device, streaming_source=None):
    import inspect
    config = json.loads((Path(path) / "config.json").read_text())
    assert streaming_source is None or config["model_type"] == "mimi"
    extra_sources = []
    if config["model_type"] == "qwen3_tts_tokenizer_12hz":
        import importlib.util
        import sys
        package = importlib.util.find_spec("qwen_tts")
        # Import the installed 12 Hz namespace without the unrelated 25 Hz audio dependencies.
        sys.path.insert(0, str(Path(next(iter(package.submodule_search_locations))) / "core"))
        from tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
        model, loading = Qwen3TTSTokenizerV2Model.from_pretrained(
            path, dtype=torch.float32, attn_implementation="eager", local_files_only=True,
            output_loading_info=True)
        assert not any(loading.values()), f"incomplete reference weights: {loading}"
        model = model.to(device).eval()
        meta = dict(frame_codes=config["encoder_valid_num_quantizers"],
                    frame_samples=config["decode_upsample_rate"], sample_rate=config["output_sample_rate"])
        d = config["decoder_config"]
        assert meta["frame_codes"] == d["num_quantizers"]
        meta["codebook_sizes"] = ([d["semantic_codebook_size"]] * d["num_semantic_quantizers"]
                                  + [d["codebook_size"]] * (d["num_quantizers"] - d["num_semantic_quantizers"]))

        def decode(codes):
            values = model.decode(audio_codes=torch.from_numpy(codes)[None].to(device),
                                  return_dict=True).audio_values[0]
            return values.float().cpu().numpy()
    elif config["model_type"] == "mimi":
        from transformers import MimiModel
        model, loading = MimiModel.from_pretrained(
            path, dtype=torch.float32, attn_implementation="eager", local_files_only=True,
            output_loading_info=True)
        assert not any(loading.values()), f"incomplete reference weights: {loading}"
        model = model.to(device).eval()
        model.config.use_cache = model.decoder_transformer.config.use_cache = False
        meta = dict(frame_codes=model.config.num_quantizers,
                    frame_samples=round(config["sampling_rate"] / config["frame_rate"]),
                    sample_rate=config["sampling_rate"])
        meta["codebook_sizes"] = [config["codebook_size"]] * meta["frame_codes"]

        if streaming_source is None:
            def decode(codes):
                values = model.decode(audio_codes=torch.from_numpy(codes.T.copy())[None].to(device),
                                      return_dict=True).audio_values[0, 0]
                return values.float().cpu().numpy()
            meta["reference_kind"] = "hf_eager_full_history"
        else:
            import importlib.util
            spec = importlib.util.spec_from_file_location("mimi_streaming_reference", streaming_source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            stream = module.StaticMimi(model, max_slots=1, batch_buckets=(1,))
            slot = torch.zeros(1, dtype=torch.long, device=device)

            def decode(codes):
                stream.reset(0, zero_kv=True)
                frames = [stream._forward(slot, torch.from_numpy(frame.copy())[None, :, None].to(device))[0, 0]
                          for frame in codes]
                return torch.cat(frames).float().cpu().numpy()
            meta["reference_kind"] = "upstream_static_streaming"
            meta["attention_window_rows"] = stream.window
            extra_sources.append(str(Path(streaming_source).resolve()))
    else:
        raise ValueError(f"unsupported reference tokenizer: {config['model_type']}")
    meta["source_files"] = [inspect.getfile(type(model)), inspect.getfile(type(model.config)), *extra_sources]
    return decode, meta


@torch.inference_mode()
def decoder_reference(a):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm"))
    from fp32_ref_gate import file_sha

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    reference = json.loads(Path(a.texts).read_text())
    assert {"config.json", "model.safetensors"} <= reference["source_sha256"].keys()
    inputs = [Path(a.texts), Path(__file__), Path(file_sha.__code__.co_filename)]
    for name, expected in reference["source_sha256"].items():
        path = Path(a.model) / name
        assert file_sha(path) == expected, f"changed tokenizer source: {path}"
        inputs.append(path)
    decode, meta = load_reference_decoder(a.model, a.device, getattr(a, "mimi_streaming_source", None))
    for name in ("frame_codes", "frame_samples", "sample_rate"):
        assert meta[name] == reference[name], f"decoder {name} differs from saved reference"
    inputs.extend(Path(p) for p in meta["source_files"])
    sources = {c["case"]: c for c in reference["cases"]}
    assert len(sources) == len(reference["cases"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    decoded, records = {}, []
    for result_path in map(Path, a.decoder_result):
        native = json.loads(result_path.read_text())
        assert native["whole_relative_l2_limit"] == 1e-5
        assert native["checks"], "empty decoder capture"
        pcm_path = result_path.with_suffix(".f32")
        assert pcm_path.stat().st_size % 4 == 0
        pcm = np.fromfile(pcm_path, dtype="<f4")
        inputs.extend([result_path, pcm_path])
        owned = None
        if "owned_whole_pcm" in native:
            owned_path = Path(native["owned_whole_pcm"])
            assert owned_path.stat().st_size % 4 == 0
            owned = np.fromfile(owned_path, dtype="<f4")
            inputs.append(owned_path)
        offset, owned_offset, checks = 0, 0, []
        for item in native["checks"]:
            source, prefix, fresh = item["source"], item["reference_frames"], item["fresh_frames"]
            case = sources[source]
            assert prefix >= 0 and fresh > 0 and prefix + fresh <= case["frames"]
            count = fresh * meta["frame_samples"]
            assert item["samples"] == count and item["output_offset_f32"] == offset
            assert offset + 3 * count <= pcm.size, "truncated native PCM"
            key = (source, prefix + fresh)
            if key not in decoded:
                codes_path = Path(case["codes"])
                assert codes_path.stat().st_size == case["frames"] * meta["frame_codes"] * 4
                codes = np.fromfile(codes_path, dtype="<u4").reshape(case["frames"], meta["frame_codes"])
                codes = codes[:prefix + fresh].astype(np.int64)
                assert np.all(codes < np.asarray(meta["codebook_sizes"])[None]), "invalid reference code"
                whole = np.asarray(decode(codes), dtype=np.float32)
                assert whole.shape == ((prefix + fresh) * meta["frame_samples"],)
                assert np.isfinite(whole).all(), "nonfinite official decoder output"
                assert np.array_equal(whole, decode(codes)), "nonrepeatable official decoder output"
                path = out / f"reference-{len(decoded)}.f32"
                whole.astype("<f4").tofile(path)
                decoded[key] = (whole, path)
                inputs.append(codes_path)
            whole, path = decoded[key]
            expected = whole[prefix * meta["frame_samples"]:].astype(np.float64)
            metrics = {}
            for name in ("whole", "stateful", "primed"):
                actual = pcm[offset:offset + count].astype(np.float64)
                finite = bool(np.isfinite(actual).all())
                error = actual - expected
                relative = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-30)) if finite else None
                metrics[name] = dict(finite=finite, relative_l2=relative,
                                     max_abs=float(np.abs(error).max()) if finite else None,
                                     passed=finite and relative < native["whole_relative_l2_limit"])
                offset += count
            if owned is not None:
                full_count = (prefix + fresh) * meta["frame_samples"]
                assert item["owned_whole_offset_f32"] == owned_offset and item["owned_whole_samples"] == full_count
                assert owned_offset + full_count <= owned.size, "truncated owned whole PCM"
                actual = owned[owned_offset:owned_offset + full_count].astype(np.float64)
                finite = bool(np.isfinite(actual).all())
                error = actual - whole.astype(np.float64)
                relative = float(np.linalg.norm(error) / max(np.linalg.norm(whole.astype(np.float64)), 1e-30)) if finite else None
                metrics["owned_whole"] = dict(finite=finite, relative_l2=relative,
                                             max_abs=float(np.abs(error).max()) if finite else None,
                                             passed=finite and relative < native["whole_relative_l2_limit"])
                owned_offset += full_count
            checks.append(dict(width=item["width"], item=item["item"], source=source,
                               reference_frames=prefix, fresh_frames=fresh, reference_pcm=str(path),
                               metrics=metrics, passed=all(m["passed"] for m in metrics.values())))
        assert offset == pcm.size, "unexpected trailing native PCM"
        assert owned is None or owned_offset == owned.size, "unexpected trailing owned whole PCM"
        records.append(dict(capture=str(result_path), internal_gate_pass=native["pass"], checks=checks,
                            independent_reference_pass=all(c["passed"] for c in checks)))
    assert records, "no native decoder captures"
    import transformers
    report = dict(scope="saved codes through official FP32 decoder vs captured whole/stateful/primed PCM",
                  pass_all=all(r["internal_gate_pass"] and r["independent_reference_pass"] for r in records),
                  relative_l2_limit=1e-5, reference_repeat_exact=True,
                  full_model_qualified=False, timings_scored=False,
                  torch=torch.__version__, transformers=transformers.__version__, device=a.device,
                  decoder=meta, captures=records,
                  hashes={str(p): file_sha(p) for p in sorted(set(inputs))})
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(pass_all=report["pass_all"], decoded_inputs=len(decoded),
                         captures=len(records), report=str(out / "report.json"))), flush=True)
    return report


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
        self.talker, self.sub, self.rp = (50, 0.9), (50, 0.9), 1.05
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
                logits[t_] = logits[t_] * self.rp if logits[t_] < 0 else logits[t_] / self.rp
            l0s.append(logits.cpu())
            c0 = self.sample(logits, greedy, gen, *self.talker)
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
                c = self.sample(lg, greedy, gen, *self.sub)
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
    # Sampling (the published generation config by default; the code predictor takes no penalty).
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--sub-top-k", type=int, default=50)
    ap.add_argument("--sub-temperature", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--force", default=None)
    ap.add_argument("--prepared-predictor", action="store_true",
                    help="FP32 conditional predictor: MODEL=prepared checkpoint, TEXTS=packet predictor capture")
    ap.add_argument("--decoder-reference", action="store_true",
                    help="official FP32 decoder: MODEL=tokenizer snapshot, TEXTS=saved codec-reference JSON")
    ap.add_argument("--decoder-result", action="append", default=[],
                    help="captured priming result JSON with whole/stateful/primed PCM; repeatable")
    ap.add_argument("--mimi-streaming-source",
                    help="pinned upstream source defining StaticMimi; explicit sliding-window streaming reference")
    a = ap.parse_args()
    if a.decoder_reference:
        if a.prepared_predictor or a.dtype != "fp32" or not a.decoder_result:
            ap.error("--decoder-reference requires --dtype fp32 and --decoder-result, without --prepared-predictor")
        if not decoder_reference(a)["pass_all"]:
            raise SystemExit(1)
        return
    if a.decoder_result or a.mimi_streaming_source:
        ap.error("--decoder-result and --mimi-streaming-source require --decoder-reference")
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
    loop.talker, loop.sub, loop.rp = (a.top_k, a.temperature), (a.sub_top_k, a.sub_temperature), a.repetition_penalty
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
                do_sample=not greedy, subtalker_dosample=not greedy, top_k=a.top_k, top_p=1.0,
                temperature=a.temperature, subtalker_top_k=a.sub_top_k, subtalker_top_p=1.0,
                subtalker_temperature=a.sub_temperature, repetition_penalty=a.repetition_penalty,
                max_new_tokens=a.max_frames)
            oc = off[0].cpu().numpy().astype(np.int32)
            n = min(len(oc), len(codes))
            rec["official_frames"] = int(len(oc))
            rec["official_prefix_match"] = int(next((i for i in range(n) if not (oc[i] == codes[i]).all()), n))
        meta.write(json.dumps(rec) + "\n")
        meta.flush()
        print(json.dumps({k: rec[k] for k in rec if k != "ids"}), flush=True)


if __name__ == "__main__":
    main()
