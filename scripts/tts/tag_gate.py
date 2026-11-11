"""Shared request construction and CPU scoring for eval.py/asr_check.py speech-tag modes."""
from __future__ import annotations

import array
import hashlib
import json
import math
import statistics
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recipe_common as rc

POSITIONS = ("start", "mid", "end", "double")
LITERALS = {"paren", "curly", "plain_prefix", "asterisk_unknown"}
LENIENT_VALUES = ("auto", "strict", "off", None, "unknown-policy", "", 0, 1.25, False, True, [], {}, ["auto"], {"mode": "off"})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def entry_for(policy, name, language):
    entry = policy["map"]["tags"][name]
    while True:
        langs = entry.get("langs", [])
        if langs and language not in langs:
            return dict(drop=True)
        if "alias" not in entry:
            return entry
        entry = policy["map"]["tags"][entry["alias"]]


def native_grid(config):
    policy, settings = config["policy"], config["settings"]
    cells = {(tag, lang, position) for tag in rc.CANONICAL_TAGS for lang in settings["languages"]
             if "text" in entry_for(policy, tag, settings.get("request_languages", {}).get(lang, lang) or None) for position in POSITIONS}
    declared = {lang for e in policy["map"]["tags"].values() for lang in e.get("langs", []) if "text" in e}
    selected = {settings.get("request_languages", {}).get(lang, lang) or lang for lang in settings["languages"]}
    if declared - selected:
        raise ValueError("native sweep languages omit map languages: " + ", ".join(sorted(declared - selected)))
    return cells


def validate_config(config):
    rc.speech_tag_policy(dict(speech_tags=config["policy"]))
    settings = config["settings"]
    allowed = {"languages", "cer_max", "cer_margin", "pause_tolerance_ms", "seed", "seeds", "native_metric", "voice", "voices", "texts", "request_languages", "sample_rate", "speed", "request", "forbid"}
    if set(settings) - allowed:
        raise ValueError("unknown tts_tags setting: " + ", ".join(sorted(set(settings) - allowed)))
    if "waiver" in config["policy"]:
        raise ValueError("a waived policy is recorded by campaign, not captured as a tag pass")
    if not isinstance(settings.get("languages"), list) or not settings["languages"] or len(set(settings["languages"])) != len(settings["languages"]):
        raise ValueError("tts_tags needs unique languages")
    limit = settings.get("cer_max")
    if type(limit) not in (float, int) or not math.isfinite(limit) or limit < 0:
        raise ValueError("tts_tags needs the recipe's existing cer_max ceiling")
    for key, expected in (("cer_margin", .05), ("pause_tolerance_ms", 80)):
        if settings.get(key, expected) != expected:
            raise ValueError(f"{key}: new thresholds require an owner decision; current value is {expected}")
    if config.get("mode", "gate") not in ("gate", "native-sweep"):
        raise ValueError("unknown tag gate mode")
    if config.get("mode") == "native-sweep":
        seeds = settings.get("seeds")
        if not isinstance(seeds, list) or len(set(seeds)) < 5 or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("native sweep requires >=5 distinct nonnegative integer seeds")
        if settings.get("native_metric") != "cer":
            raise ValueError("select native_metric = 'cer' (mean CER decrease); other efficacy metrics are owner proposals")
        if not native_grid(config):
            raise ValueError("native sweep has no native tags")


def sentences(settings, language, fixture):
    text = settings.get("texts", {}).get(language)
    if text is None:
        bases = [r["reference"] for r in fixture if r["id"] == language + "-baseline"]
        if not bases:
            raise ValueError(f"tts_tags.texts must supply two neutral sentences for {language}")
        text = bases[0]
    if isinstance(text, str):
        import re
        text = [s.strip() for s in re.split(r"(?<=[.!?।。！？])\s*", text) if s.strip()]
    if not isinstance(text, list) or len(text) != 2 or any(not isinstance(s, str) or not s.strip() for s in text):
        raise ValueError(f"tts_tags.texts.{language} requires two neutral sentences")
    return text


def make_cases(config):
    validate_config(config)
    settings, policy = config["settings"], config["policy"]
    fixture = rc.load_json(Path(__file__).with_name("tag_cases.json"))
    cases = []
    for language in settings["languages"]:
        first, second = sentences(settings, language, fixture)
        plain = first + " " + second
        if any(c in plain for c in "[<*\\"):
            raise ValueError("tag gate baseline must be tag-free")
        seed = settings.get("seed", 7)
        request = dict(voice=settings.get("voices", {}).get(language, settings.get("voice", "default")),
                       language=settings.get("request_languages", {}).get(language, language),
                       reference_language=language, seed=seed, sample_rate=settings.get("sample_rate", 8000),
                       speed=settings.get("speed", 1.14), tags="auto")
        if "request" in settings:
            # Optional cloning/instructions are matched on every arm.
            extra = settings["request"]
            if set(extra) - {"ref_audio_path", "ref_text", "non_streaming_mode", "instructions"}:
                raise ValueError("tts_tags.request accepts only matched cloning/instruction fields")
            request.update(extra)

        def add(identifier, text, kind, **extra):
            case = {**request, "id": f"{language}-{identifier}", "text": text, "reference": plain, "gate_kind": kind, "expected_status": 200, **extra}
            cases.append(case)
            return case["id"]

        if config.get("mode") == "native-sweep":
            for sweep_seed in settings["seeds"]:
                base = add(f"s{sweep_seed}-baseline", plain, "sweep-baseline", seed=sweep_seed)
                for tag, lang, position in sorted(native_grid(config)):
                    if lang != language:
                        continue
                    marker = f"[{tag}]"
                    text = {"start": marker + plain, "mid": first + marker + second,
                            "end": plain + marker, "double": marker + marker + plain}[position]
                    add(f"s{sweep_seed}-{tag}-{position}", text, "native", seed=sweep_seed,
                        baseline=base, native_tag=tag, native_position=position, forbid=settings.get("forbid", {}).get(language, []))
            continue

        base = add("baseline", plain, "baseline")
        for order, mode in enumerate(("auto", "off", "off", "auto")):
            add(f"identity-{order}", plain, "identity", tags=mode, pair=f"{language}-identity")
        for i, value in enumerate(LENIENT_VALUES):
            add(f"lenient-{i}", plain, "lenient", tags=value)
        # Raw fixture shape substitutions retain markup positions and all accepted spellings.
        old_base = next(r["reference"] for r in fixture if r["id"] == "en-baseline")
        old_first, old_second = sentences({}, "en", fixture)
        for row in fixture:
            if not row["id"].startswith("en-"):
                continue
            shape = row["id"].split("-", 1)[1]
            if shape == "baseline" or shape.startswith("clone_"):
                continue
            text = row["text"].replace(old_base, plain).replace(old_first, first).replace(old_second, second)
            literal = shape in LITERALS
            rejected = any(entry_for(policy, tag, request["language"] or None).get("reject") for tag in row["canonical_tags"])
            rejected |= shape == "br_conversational" and policy["map"]["unknown"] == "reject"
            add(shape, text, "literal" if literal else "accepted", baseline=base, expected_status=400 if rejected else 200,
                forbid=[] if literal else row.get("forbid", []) + settings.get("forbid", {}).get(language, []), pair=f"{language}-literal-{shape}" if literal else "")
            if literal:
                add(shape + "-off", text, "literal", tags="off", pair=f"{language}-literal-{shape}")
        text = "*unlisted action* " + plain
        for mode in ("auto", "off"):
            add("asterisk-unknown-" + mode, text, "literal", tags=mode, pair=f"{language}-literal-asterisk-unknown")
        # Unknown bracket words are accepted controls. Native passthrough maps retain their
        # explicit policy; the capture reveals whether the model speaks them.
        add("unknown-bracket", "[unlisted action] " + plain, "accepted", baseline=base, forbid=["unlisted", "action"] + settings.get("forbid", {}).get(language, []), expected_status=400 if policy["map"]["unknown"] == "reject" else 200)
        for position in ("leading", "internal", "trailing"):
            pair = f"{language}-pause-{position}"
            for ms in (0, 600):
                marker = f"[pause {ms}ms]"
                text = {"leading": marker + plain, "internal": first + marker + second,
                        "trailing": plain + marker}[position]
                add(f"pause-{position}-{ms}", text, "pause", pair=pair, pause_ms=ms, pause_position=position)
        dropped = next((name for name in rc.CANONICAL_TAGS if entry_for(policy, name, request["language"] or None).get("drop")), None)
        header_text = first + (f"[{dropped}]" if dropped else "") + "[pause 600ms]" + second
        add("headers", header_text, "headers",
            expected_headers={"x-plow-tags-applied": ["pause"], "x-plow-tags-dropped": [dropped] if dropped else []})
        add("headers-off", header_text, "headers-off", tags="off")
    if len({r["id"] for r in cases}) != len(cases):
        raise ValueError("duplicate tag gate case id")
    return cases


def read_pcm(path):
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("expected mono PCM16")
        pcm = array.array("h", w.readframes(w.getnframes()))
        if sys.byteorder != "little":
            pcm.byteswap()
        return w.getframerate(), pcm


def pause_score(control, treatment, directory):
    from asr_check import signal_report
    rate, a = read_pcm(directory / control["wav"])
    other_rate, b = read_pcm(directory / treatment["wav"])
    if rate != other_rate:
        raise ValueError("pause control sample-rate mismatch")
    count = rate * treatment["pause_ms"] // 1000
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    inserted = (len(b) == len(a) + count and b[prefix:prefix + count] == array.array("h", [0]) * count
                and b[:prefix] + b[prefix + count:] == a)
    position = treatment["pause_position"]
    spec = dict(ms=treatment["pause_ms"], position=position)
    if position == "internal":
        spec["window_ms"] = [max(0, prefix * 1000 / rate - 80), prefix * 1000 / rate + 80]
    result = signal_report([s / 32768 for s in b], rate,
                          dict(pauses=[spec], duration_delta=dict(baseline=control["wav"], ms=treatment["pause_ms"])), str(directory))
    return dict(result, insertion_identity=inserted, passed=inserted and result["passed"])


def sweep_score(config, captures, asr_rows):
    by_id = {r["case_id"]: r for r in captures}
    transcribed = {r["case_id"]: r for r in asr_rows}
    cells = []
    why = []
    planned = {r["id"]: r for r in make_cases(config)}
    if set(by_id) != set(planned) or len(by_id) != len(captures) or set(transcribed) != set(planned) or len(transcribed) != len(asr_rows):
        why.append("native sweep has missing, extra or duplicate cases/transcripts")
    for identifier, wanted in planned.items():
        row, ar = by_id.get(identifier, {}), transcribed.get(identifier, {})
        if row.get("status") != 200 or row.get("error") or any(row.get(k) != v for k, v in wanted.items() if k not in ("id", "language", "ref_audio_path")):
            why.append(f"native request differs from planned case: {identifier}")
        value = ar.get("cer")
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            why.append(f"invalid native CER: {identifier}")
        elif wanted["gate_kind"] == "sweep-baseline" and value > config["settings"]["cer_max"]:
            why.append(f"native baseline exceeds recipe ceiling: {identifier}")
        elif wanted["gate_kind"] == "native":
            baseline = transcribed.get(wanted["baseline"], {}).get("cer")
            from asr_check import forbid_hits
            if type(baseline) not in (int, float) or not math.isfinite(baseline) or value > baseline + .05 or forbid_hits(ar.get("hyp", ""), wanted.get("forbid", []), wanted["reference_language"]):
                why.append(f"native control spoken or CER regressed: {identifier}")
        if (ar.get("text"), ar.get("language")) != (wanted["reference"], wanted["reference_language"]):
            why.append(f"native ASR reference differs: {identifier}")
    for tag, language, position in sorted(native_grid(config)):
        deltas, seeds = [], set()
        for row in captures:
            if (row.get("native_tag"), row.get("reference_language"), row.get("native_position")) != (tag, language, position):
                continue
            if row["seed"] in seeds:
                why.append(f"duplicate native seed {tag}/{language}/{position}")
            seeds.add(row["seed"])
            control = by_id.get(row.get("baseline"))
            candidate = transcribed.get(row["case_id"])
            base = transcribed.get(row.get("baseline"))
            if not control or not base or not candidate or control["seed"] != row["seed"]:
                why.append(f"missing matched native pair {row['case_id']}")
                continue
            deltas.append(candidate["cer"] - base["cer"])
        passed = set(config["settings"]["seeds"]) == seeds and len(deltas) == len(seeds) and len(seeds) >= 5 and statistics.mean(deltas) < 0
        cells.append(dict(tag=tag, language=language, position=position, seeds=sorted(seeds),
                          mean_cer_delta=statistics.mean(deltas) if deltas else None, passed=passed))
        if not passed:
            why.append(f"no measured CER improvement over >=5 seeds: {tag}/{language}/{position}")
    return dict(metric="mean CER decrease vs matched untagged seed", cells=cells, **{"pass": bool(cells) and not why, "why": why})


def conditioning_digest(captures):
    return digest(sorted({(row.get("reference_language", ""), row.get("voice") or "", row.get("ref_audio_sha256") or "")
                          for row in captures}))


def request_digest(config):
    return digest({k: v for k, v in config["settings"].items() if k not in ("seeds", "native_metric")})


def score(config, captures, asr_report, directory):
    from asr_check import relative_cer, forbid_hits
    validate_config(config)
    expected = make_cases(config)
    expected_by_id = {r["id"]: r for r in expected}
    why = []
    if config.get("asr_model") and asr_report.get("model") != config["asr_model"]:
        why.append("ASR model differs from captured config")
    if len(captures) != len(expected) or {r.get("case_id") for r in captures} != set(expected_by_id):
        why.append("missing, extra or duplicate capture rows")
    asr_rows = asr_report.get("rows", [])
    transcript = {r.get("case_id"): r for r in asr_rows}
    expected_audio = {r.get("case_id") for r in captures if "wav" in r}
    if set(transcript) != expected_audio or len(transcript) != len(asr_rows):
        why.append("missing, extra or duplicate ASR rows")
    baselines = dict(rows=[r for r in asr_rows if expected_by_id.get(r.get("case_id"), {}).get("gate_kind") in ("baseline", "sweep-baseline")])
    pairs = {}
    pauses = []
    for row in captures:
        identifier = row.get("case_id")
        want = expected_by_id.get(identifier)
        if want is None:
            continue
        if any(row.get(k) != v for k, v in want.items() if k not in ("id", "language", "ref_audio_path")):
            why.append(f"request metadata differs from planned case: {identifier}")
        if row.get("status") != want["expected_status"] or row.get("error"):
            why.append(f"request failed: {identifier}")
            continue
        if want["expected_status"] != 200:
            if "wav" in row or not row.get("expected_error"):
                why.append(f"rejected control returned audio: {identifier}")
            continue
        if "wav" not in row:
            why.append(f"request returned no audio: {identifier}")
            continue
        if not (directory / row["wav"]).is_file():
            why.append(f"missing audio: {identifier}")
            continue
        if row.get("pair"):
            pairs.setdefault(row["pair"], []).append(row)
        headers = row.get("response_headers", {})
        if row["gate_kind"] == "headers":
            for name in ("x-plow-tags-applied", "x-plow-tags-dropped"):
                if name not in headers:
                    why.append(f"missing response header {name}")
            for name, terms in row["expected_headers"].items():
                if set(filter(None, headers.get(name, "").split(","))) != set(terms):
                    why.append(f"wrong {name}: {identifier}")
        if row["gate_kind"] == "headers-off" and any(k.startswith("x-plow-tags-") for k in headers):
            why.append("tags=off returned tag report headers")
        ar = transcript.get(identifier)
        if not ar:
            continue
        if (ar.get("text"), ar.get("language")) != (row["reference"], row["reference_language"]):
            why.append(f"ASR reference differs from planned case: {identifier}")
        cer = ar.get("cer")
        if type(cer) not in (float, int) or not math.isfinite(cer) or cer < 0:
            why.append(f"invalid CER: {identifier}")
            continue
        if row["gate_kind"] in ("baseline", "sweep-baseline") and cer > config["settings"]["cer_max"]:
            why.append(f"baseline CER exceeds recipe ceiling: {identifier}")
        if row["gate_kind"] in ("accepted", "native"):
            relative = relative_cer(dict(ar, baseline=row["baseline"]), baselines, .05)
            if not relative["passed"] or forbid_hits(ar["hyp"], row.get("forbid", []), row["reference_language"]):
                why.append(f"spoken control or CER regression: {identifier}")
    for name, rows in pairs.items():
        if all(r["gate_kind"] in ("literal", "identity") for r in rows):
            required = 4 if rows[0]["gate_kind"] == "identity" else 2
            if len(rows) != required or len({(read_pcm(directory / r["wav"])[0], hashlib.sha256(read_pcm(directory / r["wav"])[1].tobytes()).hexdigest()) for r in rows}) != 1:
                why.append(f"PCM identity failed: {name}")
        elif all(r["gate_kind"] == "pause" for r in rows):
            if len(rows) != 2:
                why.append(f"pause pair missing: {name}")
                continue
            control, treatment = sorted(rows, key=lambda r: r["pause_ms"])
            result = pause_score(control, treatment, directory)
            pauses.append(dict(pair=name, **result))
            if not result["passed"]:
                why.append(f"pause failed: {name}")
    native = None
    if config.get("mode") == "native-sweep":
        native = sweep_score(config, captures, asr_rows)
        why += native["why"]
    elif native_grid(config):
        path = config.get("native_sweep")
        if not path or not Path(path).is_file():
            why.append("native map requires its measured native-sweep report")
        else:
            evidence = rc.load_json(Path(path))
            if (evidence.get("map_sha256"), evidence.get("packet_sha256"), evidence.get("runtime_sha256"), evidence.get("request_sha256"), evidence.get("asr_model"), evidence.get("conditioning_sha256")) != (
                    digest(config["policy"]["map"]), config.get("packet_sha256"), config.get("runtime_sha256"), request_digest(config), config.get("asr_model"), conditioning_digest(captures)):
                why.append("native sweep belongs to a different map/packet")
            elif not evidence.get("pass") or evidence.get("mode") != "native-sweep":
                why.append("native sweep did not pass")
            else:
                native_config = dict(config, mode="native-sweep", settings={**config["settings"], **{k: evidence["sweep_settings"][k] for k in ("seeds", "native_metric")}})
                validate_config(native_config)
                native = sweep_score(native_config, evidence["captures"], evidence["asr_rows"])
                why += native["why"]
    return dict(mode=config.get("mode", "gate"), packet_sha256=config.get("packet_sha256"),
                map_sha256=digest(config["policy"]["map"]), config_sha256=digest(config),
                runtime_sha256=config.get("runtime_sha256"), asr_model=config.get("asr_model"), request_sha256=request_digest(config),
                conditioning_sha256=conditioning_digest(captures),
                n=len(captures), pauses=pauses, native=native, captures=captures, asr_rows=asr_rows,
                sweep_settings={k: config["settings"][k] for k in ("seeds", "native_metric", "languages") if k in config["settings"]},
                **{"pass": not why, "why": why})
