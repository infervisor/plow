#!/usr/bin/env python3
"""Run deterministic document tasks, warm-prefix replay, and concurrent requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1:
        ap.error("concurrency and max-tokens must be positive")
    with urllib.request.urlopen(args.url + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    policy = (
        "Service handbook: Standard orders ship within two business days. "
        "Express orders ship the same day if accepted before 14:00 UTC. "
        "Refunds are available within 30 days of delivery. Damaged items qualify "
        "for a replacement without a return. Escalate unresolved cases to the "
        "support lead. Never promise a delivery date without tracking data.\n"
    )
    prompts = [
        ("support_reply", policy * 12 + "\nCustomer: My order arrived damaged yesterday. "
         "Write a brief helpful reply using this policy."),
        ("incident_handoff", policy * 32 + "\nIncident: 12 express orders accepted at "
         "13:00 UTC remain unshipped at 18:00 UTC. Tracking is unavailable. "
         "Write a concise handoff with known facts and next actions."),
        ("field_extraction", policy * 64 + "\nRecord: order ORD-1042; customer Maya; "
         "delivered 2026-09-08; issue damaged item; requested replacement. "
         "Return JSON with order_id, issue, requested_resolution and policy_eligible."),
    ]

    def request(item, state):
        name, prompt = item
        body = {"model": model, "temperature": 0, "max_tokens": args.max_tokens,
                "stream": False, "messages": [{"role": "user", "content": prompt}]}
        start = time.monotonic()
        req = urllib.request.Request(args.url + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as response:
            result = json.load(response)
        choice = result["choices"][0]
        text = choice["message"]["content"]
        assert text and result["usage"]["completion_tokens"] > 0, result
        return {"workload": name, "state": state, "elapsed_s": time.monotonic() - start,
                "text": text, "finish_reason": choice["finish_reason"], "usage": result["usage"]}

    references = {}
    for item in prompts:
        cold = request(item, "initial")
        print(json.dumps(cold), flush=True)
        warm = request(item, "warm")
        warm["matches_initial"] = warm["text"] == cold["text"]
        references[item[0]] = warm["text"]
        cached = warm["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        assert cached > 0, (item[0], "prefix cache was not reused")
        print(json.dumps(warm), flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(request, prompts[i % len(prompts)], "concurrent")
                   for i in range(args.concurrency)]
        for future in futures:
            result = future.result()
            result["matches_warm"] = result["text"] == references[result["workload"]]
            assert result["matches_warm"], "concurrent warm output differs"
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
