#!/usr/bin/env python3
"""Exercise cross-slot prefixes, divergent suffixes, and owner cancellation over HTTP."""

import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import re
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--expect-shared", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "needle", Path(__file__).resolve().parents[4] / "scripts/glm53_needle_probe.py"
    )
    needle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(needle)
    model = json.load(urllib.request.urlopen(args.url + "/v1/models"))["data"][0]["id"]

    def request(endpoint, body):
        return urllib.request.urlopen(urllib.request.Request(
            args.url + endpoint, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ), timeout=600)

    def tokenize(text):
        with request("/tokenize", dict(model=model, prompt=text, add_special_tokens=False)) as response:
            return json.load(response)["tokens"]

    common = tokenize("\n".join(n[1] for n in needle.NEEDLES) + "\n" + needle.FILLER * 400)
    assert len(common) > 16384
    expected = ["9137", "2846", "6502"]
    suffixes = [
        "\n\nFor this request only, the verification number is " + number
        + ".\nQ: What is the verification number for this request?\nA: The number is"
        for number in expected
    ]
    suffixes.extend("\n\n" + n[2] for n in needle.NEEDLES)
    expected.extend(n[3] for n in needle.NEEDLES)
    prompts = [common + tokenize(suffix) for suffix in suffixes]
    boundary_suffix = tokenize("\n\n" + needle.NEEDLES[0][2])
    prompts.append((common * 2)[:32760 - len(boundary_suffix)] + boundary_suffix)
    expected.append(needle.NEEDLES[0][3])
    records = []

    def run(index):
        start = time.perf_counter()
        with request("/v1/completions", dict(
            model=model, prompt=prompts[index], add_special_tokens=False,
            max_tokens=24, temperature=0, ignore_eos=True,
        )) as response:
            result = json.load(response)
        text = result["choices"][0]["text"]
        cached = result["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        record = dict(case=index, text=text, usage=result["usage"], seconds=time.perf_counter() - start)
        assert expected[index].lower() in text.lower(), record
        if index < 3:
            assert re.findall(r"\b\d{4}\b", text)[0] == expected[index], record
        if args.expect_shared:
            assert 16384 <= cached < len(prompts[index]), record
        print(f"case={index} cached_tokens={cached} seconds={record['seconds']:.3f} PASS", flush=True)
        records.append(record)
        return record

    owner = request("/v1/completions", dict(
        model=model, prompt=prompts[0], add_special_tokens=False,
        max_tokens=512, temperature=0, ignore_eos=True, stream=True,
    ))
    try:
        for line in owner:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            assert payload != b"[DONE]", "owner completed before generating a token"
            if any(c.get("text") for c in json.loads(payload).get("choices", [])):
                break
        else:
            raise AssertionError("owner stream closed before its first token")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(run, range(3)))
    finally:
        owner.close()
    # Reuse after the original owner's stream is cancelled; divergent suffixes must remain private.
    for index in range(3):
        run(index)
    for index in range(3, len(prompts)):
        run(index)
    args.output.write_text(json.dumps(dict(common_tokens=len(common), cases=records), indent=2) + "\n")
    print(f"PASS: {len(records)} retrievals, common prefix {len(common)} tokens", flush=True)


if __name__ == "__main__":
    main()
