#!/usr/bin/env python3
"""Chatterbox Multilingual V3 (ResembleAI/chatterbox t3_mtl23ls_v3) -> a plow-compilable T3
checkpoint directory plus the multilingual text frontend as packet data.

  PYTHONPATH= HF_HOME=... <python with upstream chatterbox (mtl_tts v3)> \
      scripts/tts/chatterbox_mtl_prep.py OUT_DIR [--t3 t3_mtl23ls_v3.safetensors]

The T3 checkpoint is laid out as in chatterbox_prep.py (English): Llama-520M layers, speech head /
embedding / positions, host tables t3.text_emb (2454 rows) / t3.text_pos / t3.speech_bos, and voice
conditioning rows (T3CondEnc of conds.pt, computed by the reference module). tokenizer.json is the
reference MTLTokenizer vocabulary (grapheme_mtl_merged_expanded_v1.json).

The reference text frontend (mtl_tts.punc_norm + MTLTokenizer.encode) becomes generic text rules
(crates/plowrt/src/text/rules.rs) and tables, written to OUT_DIR/text_frontend.json and
OUT_DIR/text_tables/<name>; plowc embeds both in model.pkt (pipeline strings + a metadata section):
  common   punc_norm (CJK sentence enders), lowercase, NFKD, per-language rules, "[lang]" prefix,
           ' ' -> [SPACE]
  zh       pkuseg (spacy_ontonotes CRF + default.pkl merge dictionary) word segmentation, then
           Cangjie codes for category-Lo glyphs ("[cj_x]...[cj_.]", index suffix as the reference)
  ja       pykakasi kanji -> hiragana: itaiji fold, longest dictionary match (values are the
           reference's hiragana_normalize output for the segment: hira, " " before は/へ, unknown
           kanji dropped), PUA / astral characters dropped as kakasi does, NFKD
  ko       strip (NFKD already decomposes Hangul exactly as korean_normalize)
  he, ru   none: the reference skips them when dicta_onnx / russian_text_stresser are absent (the
           published package does not depend on either)
"""
import argparse, glob, json, os, shutil, unicodedata

import torch
from safetensors.torch import load_file, save_file

OVERLAY_ROWS = 512
T3_CONTRACT = dict(start_text=255, stop_text=0, start_speech=6561, stop_speech=6562, text_vocab=2454,
                   speech_vocab=8194, max_speech_tokens=1000, cfg_weight=0.5, temperature=0.8,
                   min_p=0.05, top_p=1.0, repetition_penalty=1.2, s3_valid_below=6561,
                   # mtl_tts.generate drops the last speech token's audio (decodes to ~40 ms of noise).
                   trim_tail_tokens=1)
TOKENIZER = "grapheme_mtl_merged_expanded_v1.json"
PUNC = [("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "), ("—", "-"), ("–", "-"),
        (" ,", ","), ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")]
ENDERS = ".!?-,、，。？！"


def snapshot():
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    c = glob.glob(os.path.join(hf, "hub/models--ResembleAI--chatterbox/snapshots/*/"))
    if not c:
        raise SystemExit("ResembleAI/chatterbox not found under $HF_HOME")
    return c[0]


def rules():
    """mtl_tts.punc_norm + MTLTokenizer.encode as rule lines (kind\\targs)."""
    r = ["default_if_empty\tYou need to add some text for me to talk.", "capitalize_first", "collapse_whitespace"]
    r += [f"replace\t{a}\t{b}" for a, b in PUNC]
    r += ["trim_end\t ", f"ensure_suffix\t{ENDERS}\t.", "lowercase", "nfkd", "language", "prefix\t[{lang}]",
          "replace\t \t[SPACE]"]
    return "\n".join(r)


def zh_tables(out, src):
    """pkuseg (spacy_ontonotes) segmenter + Cangjie glyph codes."""
    import numpy as np, pickle, srsly
    from spacy_pkuseg import pkuseg
    from spacy_pkuseg.config import config
    from spacy_pkuseg.feature_extractor import FeatureExtractor
    from chatterbox.models.tokenizers.tokenizer import ChineseCangjieConverter
    seg = pkuseg()  # the reference's segmenter (model dir resolved + downloaded here)
    fx = seg.feature_extractor
    n_feat, n_tag = seg.n_feature, seg.n_tag
    w = np.asarray(seg.model.w, dtype="<f8")
    assert w.size == n_feat * n_tag + n_tag * n_tag
    names = [None] * n_feat
    for f, i in fx.feature_to_idx.items():
        names[i] = f
    # A feature holding whitespace can never fire (fragments are whitespace-split); keep its slot.
    lines = [("" if (f is None or any(c.isspace() for c in f)) else f) for f in names]
    put(out, "zh.seg.features", "\n".join(lines))
    put(out, "zh.seg.weights", w.tobytes())
    put(out, "zh.seg.unigram", "\n".join(sorted(u for u in fx.unigram if u and not any(c.isspace() for c in u))))
    tags = [t for t, _ in sorted(fx.tag_to_idx.items(), key=lambda kv: kv[1])]
    put(out, "zh.seg.tags", "\n".join(tags))
    words = set()
    if seg.postprocesser.do_process:
        words = seg.postprocesser.other_words | seg.postprocesser.common_words
        assert not seg.postprocesser.common_words
    put(out, "zh.seg.merge", "\n".join(sorted(x for x in words if x and not any(c.isspace() for c in x))))
    # Node normalization (FeatureExtractor.normalize_text): keyword chars -> '&', then num/letter classes.
    norm = [f"{c}\t&" for c in FeatureExtractor.keywords]
    if config.numLetterNorm:
        norm += [f"{c}\t**Num" for c in sorted(FeatureExtractor.num) if c not in FeatureExtractor.keywords]
        norm += [f"{c}\t**Letter" for c in sorted(FeatureExtractor.letter) if c not in FeatureExtractor.keywords]
    put(out, "zh.seg.norm", "\n".join(norm))
    put(out, "zh.seg.meta", json.dumps(dict(word_min=config.wordMin, word_max=config.wordMax,
                                            word_feature=bool(config.wordFeature))))
    conv = ChineseCangjieConverter(src)
    cj = []
    for g in conv.word2cj:
        if len(g) == 1 and unicodedata.category(g) == "Lo":
            code = conv._cangjie_encode(g)
            cj.append(f"{g}\t" + "".join(f"[cj_{c}]" for c in code) + "[cj_.]")
    put(out, "zh.cangjie", "\n".join(cj))
    return ["segment_crf\tzh.seg", "map_chars\tzh.cangjie"]


def ja_tables(out):
    """pykakasi 2.3 convert + chatterbox hiragana_normalize as itaiji fold + longest-match table."""
    import pickle
    from pykakasi.kanji import JConv, Kanwa, Itaiji
    from pykakasi.scripts import IConv, K2, H2, A2, Sym2
    from pykakasi.properties import Ch
    from chatterbox.models.tokenizers.tokenizer import is_kanji
    jconv, iconv = JConv(), IConv()
    kanwa, itaiji = Kanwa()._jisyo_table, Itaiji()._itaijidict

    def earlier(c):  # classes kakasi.convert tests before the kanji dictionary
        return c in Ch.endmark or c in Ch.long_symbols or K2.isRegion(c) or H2.isRegion(c) or A2.isRegion(c) \
            or Sym2.isRegion(c)

    fold = []
    for k, v in itaiji.items():
        c = chr(k)
        v = chr(v) if isinstance(v, int) else (v or "")
        # Folding the whole text equals kakasi's fold-for-lookup when both sides are always looked up.
        if is_kanji(c) and len(v) == 1 and is_kanji(v) and not earlier(c):
            fold.append(f"{c}\t{v}")
    put(out, "ja.itaiji", "\n".join(fold))
    table, singles = {}, set()
    for first, entries in kanwa.items():
        c0 = chr(first)
        if earlier(c0) or not jconv.isRegion(c0):
            continue
        for key, vs in entries.items():
            if any(ch in "\t\n" for ch in key):
                continue
            yomi = next((y for y, con in vs if con is None), None)
            if yomi is None:
                continue
            hira = iconv.convert(key, yomi)["hira"]
            if any(is_kanji(ch) for ch in key):
                val = (" " + hira) if hira[:1] in ("は", "へ") else hira
            else:
                val = key
            table[key] = val
            if len(key) == 1:
                singles.add(key)
    # Unknown kanji (no key matches): kakasi emits (orig, hira "") and hiragana_normalize keeps "".
    for cp in range(0x4E00, 0xA000):
        c = chr(cp)
        if c not in singles and not earlier(c):
            table[c] = ""
    put(out, "ja.kanwa", "\n".join(f"{k}\t{v}" for k, v in table.items()))
    # PUA / astral characters outside every class kakasi knows are dropped.
    dropped = [cp for cp in list(range(0xF000, 0xFFFE)) + list(range(0x10000, 0x10FFFE))
               if not earlier(chr(cp)) and not jconv.isRegion(chr(cp))]
    runs = []
    for cp in dropped:
        if runs and runs[-1][1] == cp - 1:
            runs[-1][1] = cp
        else:
            runs.append([cp, cp])
    drop = [f"{a:X}-{b:X}" for a, b in runs]
    return ["map_chars\tja.itaiji", "dict_longest\tja.kanwa", "drop_chars\t" + ",".join(drop), "nfkd"]


TABLES = {}


def put(out, name, data):
    TABLES[name] = data.encode() if isinstance(data, str) else data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--t3", default="t3_mtl23ls_v3.safetensors")
    ap.add_argument("--exaggeration", type=float, default=0.5)
    args = ap.parse_args()
    src = snapshot()
    os.makedirs(os.path.join(args.out, "voices"), exist_ok=True)
    sd = load_file(os.path.join(src, args.t3))
    if "model" in sd:
        sd = sd["model"][0]
    assert sd["text_emb.weight"].shape[0] == T3_CONTRACT["text_vocab"]
    out = {}
    for k, v in sd.items():
        if k.startswith("tfmr.layers.") or k == "tfmr.norm.weight":
            out["model." + k[len("tfmr."):]] = v
    bf = lambda t: t.to(torch.bfloat16).contiguous()
    out = {k: bf(v) for k, v in out.items()}
    out["model.embed_tokens.weight"] = bf(sd["speech_emb.weight"])
    out["lm_head.weight"] = bf(sd["speech_head.weight"])
    out["model.speech_pos_emb.weight"] = bf(sd["speech_pos_emb.emb.weight"])
    out["t3.text_emb.weight"] = sd["text_emb.weight"].float().contiguous()
    out["t3.text_pos_emb.weight"] = sd["text_pos_emb.emb.weight"].float().contiguous()
    out["t3.speech_bos"] = (sd["speech_emb.weight"][T3_CONTRACT["start_speech"]] +
                            sd["speech_pos_emb.emb.weight"][0]).float().contiguous()
    save_file(out, f"{args.out}/model.safetensors")

    from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
    cfg = dict(LLAMA_CONFIGS["Llama_520M"])
    cfg.pop("attn_implementation", None)
    cfg.update(architectures=["LlamaForCausalLM"], vocab_size=T3_CONTRACT["speech_vocab"],
               tie_word_embeddings=False, torch_dtype="bfloat16", bos_token_id=None, eos_token_id=None,
               chatterbox_t3=dict(overlay_rows=OVERLAY_ROWS, speech_pos_rows=int(sd["speech_pos_emb.emb.weight"].shape[0]),
                                  **T3_CONTRACT))
    json.dump(cfg, open(f"{args.out}/config.json", "w"), indent=1)
    json.dump({"eos_token_id": [T3_CONTRACT["stop_speech"]]}, open(f"{args.out}/generation_config.json", "w"))
    shutil.copy(os.path.join(src, TOKENIZER), f"{args.out}/tokenizer.json")

    import perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.mtl_tts import SUPPORTED_LANGUAGES
    t3 = T3(T3Config.multilingual())
    t3.load_state_dict(sd)
    t3.eval()
    conds = torch.load(os.path.join(src, "conds.pt"), map_location="cpu", weights_only=True)["t3"]
    cond = T3Cond(speaker_emb=conds["speaker_emb"], cond_prompt_speech_tokens=conds["cond_prompt_speech_tokens"],
                  emotion_adv=args.exaggeration * torch.ones(1, 1, 1))
    with torch.no_grad():
        rows = t3.prepare_conditioning(cond)[0].float().contiguous()
    rows.numpy().tofile(f"{args.out}/voices/default.f32")
    json.dump({"default": {"rows": int(rows.shape[0]), "exaggeration": args.exaggeration}},
              open(f"{args.out}/voices/voices.json", "w"), indent=1)

    langs = {code: [] for code in sorted(SUPPORTED_LANGUAGES)}
    langs["zh"] = zh_tables(args.out, src)
    langs["ja"] = ja_tables(args.out)
    langs["ko"] = ["strip"]
    tdir = os.path.join(args.out, "text_tables")
    os.makedirs(tdir, exist_ok=True)
    for name, data in TABLES.items():
        open(os.path.join(tdir, name), "wb").write(data)
    json.dump(dict(rules=rules(), languages={k: "\n".join(v) for k, v in langs.items()}, default_language="en",
                   tables=sorted(TABLES)), open(f"{args.out}/text_frontend.json", "w"), ensure_ascii=False, indent=1)
    print(f"wrote {args.out}: {len(out)} tensors, voice rows {tuple(rows.shape)}, {len(langs)} languages, "
          f"{len(TABLES)} text tables ({sum(len(v) for v in TABLES.values()) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
