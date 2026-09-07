#!/usr/bin/env python3
"""Gemma 4 packed serving parity, limits and cancellation through the HTTP API."""
import argparse
import concurrent.futures
import json
import urllib.error
import urllib.request


def post(url, body):
    request = urllib.request.Request(
        url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=300)


def prompt(length, seed):
    return (" " + ["hello", "world", "test", "data"][seed % 4]) * length


def generate(url, model, ids, limit):
    body = dict(model=model, prompt=ids, add_special_tokens=False, temperature=0, max_tokens=limit,
                ignore_eos=True, stream=True, stream_options={"include_usage": True})
    chunks, usage, reason, done = [], None, None, False
    with post(url, body) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            assert "error" not in event, event
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                chunks.append(choice.get("text", ""))
                reason = choice.get("finish_reason") or reason
    assert done and usage, (done, usage)
    assert usage["prompt_tokens"] == ids.count(" "), usage
    assert usage["completion_tokens"] == limit, usage
    assert reason == "length", reason
    return "".join(chunks)


def cancel(url, model, during_decode):
    body = dict(model=model, prompt=prompt(2305, 19), temperature=0,
                max_tokens=256, ignore_eos=True, stream=True, add_special_tokens=False)
    with post(url, body) as response:
        if during_decode:
            for line in response:
                if line.startswith(b"data: "):
                    event = json.loads(line[6:])
                    assert "error" not in event, event
                    if any(c.get("text") for c in event.get("choices", [])):
                        break
        # Closing the response disconnects the request during prefill or decode.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-ctx", type=int, required=True)
    args = parser.parse_args()
    cases = [(prompt(length, i), limit) for i, (length, limit) in enumerate(
        [(777, 1), (901, 3), (2305, 7), (2689, 17)])]
    expected = [generate(args.control, args.model, ids, limit) for ids, limit in cases]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for _ in range(2):
            actual = list(pool.map(lambda case: generate(
                args.candidate, args.model, *case), cases))
            assert actual == expected, "concurrent outputs differ from isolated control"
        print("PASS concurrent ragged prompts, exact output limits, slot reuse", flush=True)
        for during_decode in [False, True]:
            list(pool.map(lambda _: cancel(args.candidate, args.model, during_decode), range(4)))
            actual = list(pool.map(lambda case: generate(
                args.candidate, args.model, *case), cases))
            assert actual == expected, "cancellation changed surviving requests"
        print("PASS cancellation during prefill and decode, recovery", flush=True)
    try:
        with post(args.candidate, dict(model=args.model,
                  prompt=prompt(args.max_ctx + 1, 0), max_tokens=1)) as response:
            raise AssertionError(f"oversized prompt accepted: {response.read()!r}")
    except urllib.error.HTTPError as error:
        assert error.code in (400, 429), error.code
        detail = json.load(error)
        assert "max_ctx" in str(detail) or "context" in str(detail), detail
    assert generate(args.candidate, args.model, *cases[0]) == expected[0]
    print("PASS context rejection and recovery", flush=True)


if __name__ == "__main__":
    main()
