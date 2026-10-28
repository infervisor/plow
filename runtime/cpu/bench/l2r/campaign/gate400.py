import json, sys, urllib.request, urllib.error
url, ref = sys.argv[1], sys.argv[2]
r = json.load(open(ref))
cases = r['cases'] if isinstance(r, dict) and 'cases' in r else r
c = cases[0] if isinstance(cases, list) else next(iter(cases.values()))
ids = c['prompt_ids']
model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
body = {"model": model, "prompt": ids, "max_tokens": 8, "temperature": 0, "logprobs": 20,
        "return_tokens_as_token_ids": True, "ignore_eos": True}
req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
try:
    print(urllib.request.urlopen(req).read()[:300])
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read()[:600])
print("case", c.get('id'), "prompt len", len(ids), "max id", max(ids))
