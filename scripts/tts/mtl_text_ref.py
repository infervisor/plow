#!/usr/bin/env python3
"""Reference text ids of Chatterbox Multilingual (upstream mtl_tts: punc_norm + MTLTokenizer.encode)
for the plow text frontend gate (crates/plowrt/examples/t3_text_check.rs). CPU only.

  PYTHONPATH= HF_HOME=... <v3 venv python> scripts/tts/mtl_text_ref.py OUT.json [--corpus rows.json]

Texts: mtl_prompts (PROMPTS + MILESTONE), built-in edge cases, and an optional corpus
([{"language", "text"}], e.g. Wikipedia extracts).
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from mtl_prompts import PROMPTS, MILESTONE

EDGE = [
    ("zh", "你好！今天是2024年5月3日，温度２５℃。"), ("zh", "我在用ChatGPT和iPhone 15 Pro，价格是￥6999.00元"),
    ("zh", "請問台北車站怎麼走？"), ("zh", "“引号”与《书名号》……还有——破折号"), ("zh", "一二三四五六七八九十百千万亿"),
    ("zh", "他说：“我们明天见。”然后离开了。"), ("zh", "😀表情符号和emoji混合"), ("zh", ""),
    ("ja", "東京タワーへ行きたいです"), ("ja", "今日は晴れ、明日は雨でしょう。"), ("ja", "コーヒーを１杯ください！"),
    ("ja", "私は日本語を勉強しています。漢字はむずかしいですね"), ("ja", "ＡＢＣ株式会社の山田太郎です。"),
    ("ja", "㈱ゃゅょっ〜、ヴァイオリン🎻"), ("ja", "々の繰り返し、人々、時々"), ("ja", "彼へ手紙を書いた。"),
    ("ko", "  안녕하세요!  반갑습니다  "), ("ru", "Ёлка и ЁЖИК: «ТЕСТ» — проверка."), ("de", "STRASSE, Straße; Maß."),
    ("tr", "İstanbul ve IĞDIR şehirleri"), ("el", "ΟΔΟΣ και ΣΟΦΟΣ."), ("fr", "Élève… « cœur » d’œuvre"),
    ("ar", "العربية: ١٢٣ والأرقام 456؟"), ("hi", "क़िला और ज़मीन — १२३"), ("he", "שָׁלוֹם עוֹלָם"),
    ("en", "hello   world... it's 3:45 pm; OK?"), ("en", "a - b – c — d"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--corpus", action="append", default=[])
    args = ap.parse_args()
    import glob
    from chatterbox.mtl_tts import punc_norm
    from chatterbox.models.tokenizers import MTLTokenizer
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    src = glob.glob(os.path.join(hf, "hub/models--ResembleAI--chatterbox/snapshots/*/"))[0]
    tok = MTLTokenizer(os.path.join(src, "grapheme_mtl_merged_expanded_v1.json"))
    rows = [dict(language=l, text=t) for l, t in PROMPTS] + [dict(language=l, text=t) for l, t in MILESTONE.items()]
    rows += [dict(language=l, text=t) for l, t in EDGE]
    for c in args.corpus:
        rows += json.load(open(c))
    for r in rows:
        r["ids"] = tok.encode(punc_norm(r["text"]), language_id=r["language"])
    json.dump(rows, open(args.out, "w"), ensure_ascii=False)
    print(f"{len(rows)} texts -> {args.out}")


if __name__ == "__main__":
    main()
