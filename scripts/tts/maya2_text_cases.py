#!/usr/bin/env python3
"""Write the maya2_text parity corpus: (lang, text, expected) JSONL from the Python Maya 2 Native normaliser.

    PYTHONPATH=<maya2 checkout>/src python scripts/tts/maya2_text_cases.py \
        crates/plowrt/tests/data/maya2_text_cases.jsonl

`expected` is `maya3.text.normalise_text(text, lang)` with default arguments. The Rust port
(crates/plowrt/src/tts/maya2_text) must reproduce every line byte for byte (`tests::parity`). Cases are deterministic
(fixed seeds), so regenerating with the same normaliser gives the same file.
"""
import contextlib
import io
import json
import random
import sys

from maya3.text import normalise_text
from maya3.text.tn.normalize import language_table

LANGS = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

README = [
    ("en", "Your OTP is 482913 [laughs]"), ("en", "15-08-25"), ("en", "9/9/99"), ("en", "Card expiry 11/26"),
    ("en", "FY 2023-24"), ("en", "4111 2233 4455 6677"), ("en", "98765 43210"), ("hi", "पिन 4532"),
    ("en", "Deliver to 560038 please"), ("en", "Update to v4.5.2 now"), ("ta", "ஆகஸ்ட் 15, 2025"),
    ("bn", "₹95.41"), ("en", "₹95.41"), ("hi", "सात, [curious] छह"), ("hi", "write to a-b@x-y.com"),
    ("as", "বাকী 30 মিনিট"), ("bn", "40 km/h"), ("bn", "visit e-receipt.hdfcbank.com"),
    ("en", "book at irctc.co.in."), ("en", "pay rahul.deshmukh@okaxis"), ("bn", "sbi.co.in"),
    ("bn", "hdfcbank.com"), ("en", "3.5%-4%"), ("en", "₹500/kg"), ("en", "8.5% p.a."), ("hi", "8.5% p.a."),
    ("as", "পঞ্চাশ টকা ৰ"), ("as", "50 টকাৰ"), ("as", "10:30 বজাত"),
]

NUMBERS = [0, 1, 2, 5, 9, 10, 11, 12, 15, 19, 20, 21, 29, 30, 42, 45, 50, 69, 71, 99, 100, 101, 110, 115, 199, 200, 250,
           500, 999, 1000, 1001, 1010, 1100, 1947, 1999, 2000, 2001, 2025, 2026, 9999, 10000, 10001, 12000, 54000,
           99999, 100000, 100001, 101000, 123456, 250000, 999999, 1000000, 1234567, 9999999, 10000000, 10000001,
           12345678, 99999999, 100000000, 123456789, 999999999, 1000000000]

EN = {
    "numbers": ["The value is {n}.", "{n} people came", "about {n}", "Total: {n}"],
    "id": ["OTP is {d}", "Your account number {d}", "PIN {d}", "call {d} now", "order number {d}", "ref {d}",
           "ticket {d} confirmed", "pnr {d}", "{d} is your OTP", "dial {d}", "card ending in {d}"],
}


def native_frames(L):
    def pick(seq, k=3):
        seq = list(seq)
        return seq[:k] + seq[-k:] if len(seq) > 2 * k else seq
    return {
        "markers": pick(L.TIME_MARKERS, 4), "periods": pick(L.PERIOD_DETECT, 4), "years": pick(L.YEAR_WORDS, 3),
        "ids": pick(L.ID_KEYWORDS, 5), "after": pick(L.ID_AFTER_WORDS, 3), "labels": pick(L.LABEL_WORDS, 3),
        "cur": pick([w for w, k in L.CURRENCY_WORDS.items()], 4), "scale": pick(L.SCALE_WORDS, 4),
        "ord": list(L.ORDINAL_INPUT_SUFFIXES), "months": list(L.MONTHS) + list(L.MONTH_ALIASES),
        "minutes": pick(L.MINUTE_WORDS, 2),
        "pp": sorted(set(getattr(L, "_PP", ())) | set(getattr(L, "_CASE", ())) | set(getattr(L, "_PP_GLUE", ()))
                     | set(getattr(L, "_PP_SPACED", ())) | {"এ", "এর", "র", "তে", "মধ্যে", "থেকে", "ৰ", "ত", "লৈ",
                                                            "ന്റെ", "ൽ", "ിൽ", "ക്ക്", "ലേക്ക്", "ാണ്",
                                                            "க்கு", "இல்", "ல்", "ki", "lo", "ga"}),
        "nouns": sorted(getattr(L, "_ATTACH_NOUNS", ())) + sorted(getattr(L, "_COUNT_NOUNS", ()))[:6],
    }


def gen(lang, rng):
    L = language_table(lang)
    F = native_frames(L)
    ch = rng.choice
    out = []
    add = out.append

    # numbers, alone and in frames; Indian / western grouping; decimals; negatives
    for n in NUMBERS:
        add(str(n))
        if rng.random() < 0.5:
            add(ch(EN["numbers"]).format(n=n))
    for n in rng.sample(NUMBERS, 12):
        s = f"{n:,}"
        add(f"total {s}")
        ind = str(n)
        if len(ind) > 3:
            head, tail = ind[:-3], ind[-3:]
            parts = []
            while len(head) > 2:
                parts.insert(0, head[-2:]); head = head[:-2]
            if head:
                parts.insert(0, head)
            add(",".join(parts + [tail]) + " " + ch(F["cur"] or ["units"]))
    for s in ["3.14", "0.5", "12.05", "1,234.56", "-5", "-12.5", "−7", "+5", "007", "0", "00", "1.50", "100.001",
              "1,00,000", "12,34,567", "1,000,000", "1,2,3", "12,3456", "99.99", "5.", ".5", "1.2.3"]:
        add(f"value {s} here")
    for n in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 21, 22, 23, 31, 100, 101, 102, 111, 112, 1000, 1001, 100000]:
        add(f"the {n}{ch(['st', 'nd', 'rd', 'th', 'TH'])} time")
        for suf in F["ord"][:6]:
            add(f"{n}{suf}")
            add(f"{n}-{suf} {ch(F['nouns'] or ['x'])}")

    # currency
    cur = ["₹500", "₹95.41", "₹95.00", "₹0.50", "Rs. 1,250", "Rs 50", "Rs.50", "INR 2000", "$84.20", "$5", "€15",
           "£3.50", "₹5 crore", "₹2.5 lakh", "₹3 Cr", "$1.5 million", "₹2 bn", "₹500-700", "$10-$20", "₹1,000–2,000",
           "-₹50", "+₹50", "₹-50", "-₹-50", "₹ 1,00,000/-", "500/-", "1,250.50/-", "50 paise", "100 rupees", "5 dollars",
           "₹5 lakhs", "₹12 thousand", "Rs.5K", "₹10 lakh crore", "USD 300", "₹1.005", "₹ 7", "price ₹ 99.9",
           "₹50 paise", "₹5%", "$1,23,456.78"]
    for c in cur:
        add(c)
        if rng.random() < 0.35:
            add(f"pay {c} today")
    for w in F["cur"]:
        add(f"{rng.randint(2, 99999)} {w}")
        add(f"{rng.randint(2, 999)}.{rng.randint(10, 99)} {w}")
    for w in F["scale"]:
        add(f"₹{rng.randint(1, 99)} {w}")
        add(f"₹{rng.randint(1, 9)}.{rng.randint(1, 9)} {w}{ch(['चा', 'ांचा', 'ची', 'त', ''])}")
    for noun in F["nouns"][:5]:
        for pp in rng.sample(F["pp"], min(3, len(F["pp"]))):
            add(f"₹{rng.randint(1, 500)} {pp}")
            add(f"{rng.randint(1, 60)}% {pp}")
            add(f"{rng.randint(1, 60)} {noun} {pp}")
            add(f"{rng.randint(1, 60)} {noun}-{pp}")
    for pp in F["pp"]:
        add(f"{rng.randint(1, 12)}:{rng.choice(['00', '30', '45'])} {pp}")
        add(f"{rng.randint(1, 99)}-{pp}")

    # dates
    dates = ["15-08-2025", "15/08/2025", "15.08.2025", "2025-08-15", "2025-8-5", "15-08-25", "05-11-24", "9/9/99",
             "1/2/03", "8-2-45", "12/12/12", "31/12/50", "1/1/51", "15 Aug 2025", "15th August 2025", "August 15, 2025",
             "Aug 15", "12-Aug-2025", "15-16 March", "15-16 March 2024", "31/02/2024", "29/02/2023", "29/02/2024",
             "2024-02-30", "2024-13-01", "32/01/2020", "1st Jan", "3rd of May", "May 3rd, 2021", "Sept 9", "Dec 25th",
             "on 25/12", "date 12/5", "15-08-202", "1990s", "90s", "2020s", "'80s", "in 1947", "since 2001",
             "year 2030", "by 1999", "1600", "1947 में", "1422 गांधी मार्ग", "2210 MG road", "x-12/05/2024",
             "12/05/2024-x", "2024/05/12", "1-2-2024", "01.01.2001"]
    for d in dates:
        add(d)
        if rng.random() < 0.35:
            add(f"due on {d}")
    for m in F["months"]:
        add(f"{rng.randint(1, 28)} {m} {rng.randint(1990, 2030)}")
        add(f"{m} {rng.randint(1, 31)}, {rng.randint(1990, 2030)}")
        add(f"{rng.randint(1, 28)}-{m}")
        add(f"{rng.randint(1, 28)}{ch(F['ord'] or ['th'])} {m}")
    for y in [1947, 1999, 2023, 2026, 1857]:
        for w in F["years"]:
            add(f"{y} {w}")
            add(f"{w} {y}")
        add(f"{y}-{(y + 1) % 100:02d}")
        add(f"FY {y}–{(y + 1) % 100:02d}")
        add(f"{y}-{(y + 3) % 100:02d}")
    for t in ["ஆகஸ்ட் 15, 2025", "ஆகஸ்ட் 15ஆம் தேதி", "ஜனவரி 1", "மே 31 2024", "5ஆம்", "45ஆம்", "15ம் தேதி",
              "தேதி 12. 2024", "டிசம்பர் 25, 1999", "பிப்ரவரி 30"]:
        add(t)
    for e in ["Card expiry 11/26", "valid thru 08/2027", "CVV 123 exp 09/28", "expires 1/30", "date 25/12",
              "date 05/30", "मुदत 11/26", "समाप्ति 03/29", "তারিখ 07/31", "11/26 is fine", "ratio 2/3"]:
        add(e)

    # times
    times = ["10:30", "10:30 am", "10:30 AM", "6.45 pm", "7 PM", "7 p.m.", "12 am", "12:00", "00:15", "14:30", "23:59",
             "24:00", "9:60", "13 pm", "11 AM – 3 PM", "9:00-17:00", "9am-5pm", "10 AM to 2 PM", "1:05:30", "0:45:00",
             "12:75:00", "ratio 3:2", "16:9", "3:15 ratio", "odds 5:30", "at 5 o'clock", "morning 9:15",
             "8:30 evening", "4:5", "100:1", "10:30:45 hrs"]
    for t in times:
        add(t)
        if rng.random() < 0.35:
            add(f"meet at {t} sharp")
    for mk in F["markers"]:
        h = rng.randint(1, 12)
        add(f"{h} {mk}")
        add(f"{h}:{rng.randint(10, 59)} {mk}")
        add(f"{rng.randint(13, 23)}:{rng.choice(['00', '05', '30'])} {mk}")
        for pw in F["periods"][:3]:
            add(f"{pw} {h} {mk}")
            add(f"{pw} {h}:{rng.randint(10, 59)}")
        for mw in F["minutes"]:
            add(f"{h} {mk} {rng.randint(1, 59)} {mw}")
    for pw in F["periods"]:
        add(f"{pw} 7:{rng.randint(10, 59)}")
        add(f"{rng.randint(1, 11)} {rng.choice(['am', 'pm'])} {pw}")

    # percent / ranges / maths
    for p in ["50%", "12.5%", "-3%", "40-70%", "40–70%", "3.5%-4%", "3.5 % - 4 %", "10-20", "20-10 days", "20-10",
              "3-2", "50-50", "5–10", "8.5% p.a.", "8 % P.A.", "<500", ">= 18", "≤ 5%", "x > 5", "₹<500", "a<b",
              "2 < 3", "1,00,000 > 50,000", "100%", "0.5%", "1,234.5%", "10^23", "x^2", "5²", "a²", "√16",
              "E = mc^2", "x + 5 = 12", "17 × 3 = 51", "10 / 2 = 5", "a squared plus b", "2*3", "3.4K", "1.2M",
              "5B", "1920x1080", "24x7", "6 x 4", "0x1F", "₹4,583.33 x 24", "Class XII", "World War II", "Part IV",
              "MIX", "2/7", "1/2", "3/4", "1/4", "10/20", "22/7", "± 5", "25°C", "98.6°F", "90°"]:
        add(p)
        if rng.random() < 0.35:
            add(f"about {p} more")

    # identifiers
    ids = ["482913", "4471", "560038", "Deliver to 560038", "pincode 560038", "PIN 4532", "pin code: 110001",
           "Call 98765 43210", "+91 98765 43210", "1800 123 4567", "+91-9876543210", "9876543210", "+44 20 7946 0958",
           "card 4111 2233 4455 6677", "4111-2233-4455-6677", "account number 123456789012", "ending in 7104",
           "139 dial करें", "order #12345", "#4471", "ref no. 4471", "No. 12", "no. 7", "Nos. 5", "Room 314",
           "Hall 3-A", "Seat 23A", "coach B4", "Platform 5", "Gate C", "Gate C-12", "train 12951", "PNR 4523891023",
           "Flat No. 12-5-24", "House No. 8-2-45", "CUST-567890", "#560038", "560038 rupees", "₹560038",
           "560038.50", "5,60,038", "560038%", "560038 * 2", "OTP: 0042", "code 22377", "IMEI 356938035643809",
           "2345 6789 0123", "98765-43210", "12-34-56", "123-4567", "1234-56", "12345 678", "1 2 3 4 5 6 7 8 9 0",
           "aadhaar 1234 5678 9012", "UPI 9876543210", "011-23456789", "07104", "0", "000"]
    for d in ids:
        add(d)
        if rng.random() < 0.35:
            add(f"{d} please")
    for kw in F["ids"]:
        add(f"{kw} {rng.randint(1000, 999999)}")
        add(f"{kw} {rng.randint(10 ** 6, 10 ** 10)}")
        add(f"{kw}: {rng.randint(100, 9999)}-{rng.randint(100, 9999)}")
    for kw in F["after"]:
        add(f"{rng.randint(100, 99999)} {kw}")
    for lw in F["labels"]:
        add(f"{lw} {rng.randint(1, 400)}")
        add(f"{lw} {rng.randint(1, 40)}-{ch('ABC')}")
    for kw in ["phone", "mobile", "OTP", "otp", "card", "account", "ओटीपी", "ఓటీపీ"]:
        add(f"{kw} {rng.randint(100, 999999)}")

    # codes, versions, units
    for c in ["v4.5.2", "V2.0", "v10", "version 1.2", "v1.2.3.4.5", "MH12AB8821", "77K2L9", "WH-1000XM5",
              "Sony WH-1000XM5", "Kepler-186f", "Soyuz T-11", "Salyut-7", "XPS 15", "ABC 12", "OTP 12", "5G", "A4",
              "B12", "Y2K", "S24", "A380", "COVID-19", "AP 09 CL 4421", "TS-09-EA-1234", "H2O", "CO2", "NaCl", "CO",
              "NO", "HCl", "C6H12O6", "12-Aug-2025", "MAY-17Y7", "CR-2A", "3-A", "900-point", "80-watt", "30-DAY",
              "24-HOUR", "15-GB", "7-East", "A/C", "a/c no 12", "1980s", "250ml", "1.5 GB", "5,000 mAh",
              "40 km/h", "40 KM/H", "100 kmph", "60 mph", "5 kg", "5 KG", "12 cm", "10 MB/s", "5 V", "5 v", "220 W",
              "₹500/kg", "₹1200/night", "$20/month", "Rs. 45/day", "₹60/l", "₹30/gm", "-5 kg", "2.5kWh", "3000 RPM",
              "500 mg", "10 mcg", "1 km", "1 kg", "1 hr", "2 hrs", "5 mins", "45 min", "30 secs", "10 kms",
              "approx. 5", "about 5 approx.", "Dr. Rao", "Mr. Shah", "Prof Iyer", "ABC Pvt Ltd", "ETA 5 mins"]:
        add(c)
        if rng.random() < 0.35:
            add(f"see {c} now")

    # URLs / e-mails / UPI / handles
    for u in ["www.example.com", "https://maya.ai/docs?id=12&x=y", "http://a-b.co.in/path_1", "maya.ai",
              "rahul.sharma@gmail.com", "rahul.deshmukh@okaxis", "rahul@ybl", "sbi.co.in", "onlinesbi.sbi",
              "e-receipt.hdfcbank.com", "irctc.co.in.", "netbanking.nationalbank.in", "support-team@my-company.co.in",
              "kyc_update@icicibank.com", "@test_user", "@test_user.", "x@y.z", "user123@mail9.org",
              "https://www.uidai.gov.in/en/my-aadhaar", "epfindia.gov.in", "pnbindia.in", "lic.co.in",
              "https://example.com/Dr/No/1", "WWW.EXAMPLE.COM", "mutualfund.sbimf.com", "idbibank.in",
              "hdfcbank.com/kyc", "abc.com,", "(see abc.com)", "a.b@c", "foo@bar"]:
        add(u)
        if rng.random() < 0.35:
            add(f"visit {u} today")

    # caps, tags, punctuation, scripts
    for c in ["DEVANSH paid", "Mr. RAMESH KUMAR", "KAVYA", "HDFC", "NEFT", "OTP", "PRIYA SHARMA", "AADHAAR", "MUMBAI",
              "AMIT", "SBI ATM", "ANITA DESAI", "IOUA", "BCDF", "ÉCOLE", "ABCDé"]:
        add(c)
        if rng.random() < 0.35:
            add(f"{c} sent 500")
    for c in ["Hello [laughs] 5 apples", "सात, [curious] छह", "[sighs] OTP 1234 [whispers]", "[Laughs] uppercase",
              "text [two words] more", "[laughs]", "a [x] b [y]", "5,[sighs]6", "  spaced  [laughs]  out  ",
              "[123] numbers", "end with tag 42 [laughs]", "[sighs]   ", "unterminated [tag"]:
        add(c)
    for c in ['He said "5 apples"', "'Premium Home Decor' 5", "reply 'Y'", "D'Souza paid 50", "wait... 5",
              "wait…5", "(about 5)", "### Header 5", "//comment// 12", "😀 5 ⭐", "→ 10", "a • b", "1 — 2",
              "11 AM — 3 PM", "x – y", "tab\t\t5", "line\n5", "ctrl\x075", "pua5", "zero​width 5",
              "{5}", "<b>5</b>", "${x}", "a < b", "John's 2", "it's 5'", "‘quoted’ 5", ",,, 5", "5,", "5, , 6",
              "हैं 5", "क़िला 5", "x ,y", "a ; b", "5 : 6", "e-mail me", "end.", "", "   ", "a    b"]:
        add(c)
    digits = {"hi": "१२३४५", "mr": "१२३४५", "bn": "১২৩৪৫", "as": "১২৩৪৫", "gu": "૧૨૩૪૫", "pa": "੧੨੩੪੫",
              "or": "୧୨୩୪୫", "ta": "௧௨௩௪௫", "te": "౧౨౩౪౫", "kn": "೧೨೩೪೫", "ml": "൧൨൩൪൫", "en": "12345"}
    for code, ds in digits.items():
        add(f"{ds} {ch(F['cur'] or ['x'])}")
        add(f"{ds[:2]}:{ds[2:4]}")
        add(f"₹{ds}.{ds[:2]}")
    for i in range(3):
        add(f"{ch(F['ids'] or ['x'])} {ch(['482913', '9876543210', '1234'])} and {ch(F['months'])} {i + 5}")

    # random sentences mixing the shapes above
    pools = [
        lambda: f"{rng.randint(0, 10 ** rng.randint(1, 9))}",
        lambda: f"₹{rng.randint(1, 99999)}{ch(['', '.' + str(rng.randint(10, 99)), ' lakh', ' crore', '/-'])}",
        lambda: f"{rng.randint(1, 28)}/{rng.randint(1, 12)}/{rng.choice([2024, 2025, 99, 5])}",
        lambda: f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d} {ch(['am', 'pm', '', 'AM'])}",
        lambda: f"{rng.randint(1, 99)}{ch(['%', '.5%', ' km', ' kg', 'GB', 'th', 'st'])}",
        lambda: f"{ch(['OTP', 'PIN', 'account', 'phone', 'code'])} {rng.randint(100, 10 ** 10)}",
        lambda: ch(["www.test.in", "a.b@okicici", "user@gmail.com", "v2.1.3", "MH12AB1234", "[laughs]", "RAJESH",
                    "H2O", "FY 2024-25", "40 km/h", "₹20/kg", "5.5% p.a.", "12-15%", "No. 4", "Room 12"]),
        lambda: ch(F["ids"] + F["markers"] + F["years"] + F["cur"] + F["labels"] or ["x"]),
        lambda: ch(F["months"] + F["periods"] or ["x"]),
        lambda: ch(["and", "the", "to", "है", "को", "এবং", "மற்றும்", "మరియు", "ಮತ್ತು", "ഒപ്പം", "आणि", "ਅਤੇ", "ଏବଂ", "અને",
                    ",", ".", "?", "!", "-", "(", ")", ":"]),
    ]
    for _ in range(35):
        add(" ".join(ch(pools)() for _ in range(rng.randint(3, 9))))
    return out


def main():
    path = sys.argv[1]
    rows, seen = [], set()
    cases = list(README)
    for i, lang in enumerate(LANGS):
        cases += [(lang, t) for t in gen(lang, random.Random(1000 + i))]
    for lang, text in cases:
        if (lang, text) in seen:
            continue
        seen.add((lang, text))
        with contextlib.redirect_stdout(io.StringIO()):     # the normaliser's [obs] fallback lines
            expected = normalise_text(text, lang)
        rows.append(json.dumps({"lang": lang, "text": text, "expected": expected}, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"{len(rows)} cases -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
