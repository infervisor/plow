import argparse
import concurrent.futures
import json
import pathlib
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:18680')
parser.add_argument('--output', default='cpu-http-decode.json')
args = parser.parse_args()
base = args.url.rstrip('/')
with urllib.request.urlopen(base + '/v1/models') as response:
    model = json.load(response)['data'][0]['id']
questions = [
    'Explain how a refrigerator transfers heat. Use about 150 words.',
    'Explain why the sky appears blue. Use about 150 words.',
    'Explain how a bicycle stays balanced while moving. Use about 150 words.',
    'Explain how plants turn sunlight into stored energy. Use about 150 words.',
]

def run(question):
    body = {'model': model, 'messages': [{'role': 'user', 'content': question}],
            'stream': True, 'stream_options': {'include_usage': True},
            'temperature': 0, 'max_tokens': 128}
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    first, usage, finish, done = None, None, None, False
    pieces = []
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                done = True
                break
            item = json.loads(raw)
            assert 'error' not in item, item
            usage = item.get('usage') or usage
            for choice in item.get('choices', []):
                content = choice.get('delta', {}).get('content') or ''
                if content and first is None:
                    first = time.monotonic() - start
                pieces.append(content)
                finish = choice.get('finish_reason') or finish
    elapsed = time.monotonic() - start
    assert done and first is not None and usage and finish in ('stop', 'length')
    assert usage['completion_tokens'] >= 64, usage
    result = {'prompt': question, 'text': ''.join(pieces), 'seconds': elapsed,
              'ttft_seconds': first, 'finish_reason': finish, 'usage': usage,
              'mean_post_first_token_ms': 1000*(elapsed-first)/(usage['completion_tokens']-1)}
    print(json.dumps(result), flush=True)
    return result

start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(run, questions))
elapsed = time.monotonic() - start
summary = {'model': model, 'concurrency': 4, 'seconds': elapsed,
           'output_tokens_per_second': sum(r['usage']['completion_tokens'] for r in results)/elapsed,
           'results': results}
pathlib.Path(args.output).write_text(json.dumps(summary, indent=2) + '\n')
print('PASS total_seconds', elapsed, 'output_tokens_per_second', summary['output_tokens_per_second'], flush=True)
