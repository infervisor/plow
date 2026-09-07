import argparse
import concurrent.futures
import json
import pathlib
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:18680')
parser.add_argument('--output', default='cpu-http-e2e.json')
args = parser.parse_args()
BASE = args.url.rstrip('/')
results = []

def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=600)

with request('/healthz') as response:
    assert response.status == 200
with request('/v1/models') as response:
    models = json.load(response)
model = models['data'][0]['id']
print('model', model, flush=True)

def check(label, question, expected, stream=False):
    body = {'model': model, 'messages': [{'role': 'user', 'content': question}],
            'max_tokens': 32, 'temperature': 0, 'stream': stream}
    if stream:
        body['stream_options'] = {'include_usage': True}
    start = time.monotonic()
    first = None
    with request('/v1/chat/completions', body) as response:
        if stream:
            pieces, usage, done, finish = [], None, False, None
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
                    piece = choice.get('delta', {}).get('content') or ''
                    if piece and first is None:
                        first = time.monotonic() - start
                    pieces.append(piece)
                    finish = choice.get('finish_reason') or finish
            assert done and finish == 'stop' and usage, (done, finish, usage)
            answer = ''.join(pieces)
        else:
            item = json.load(response)
            assert 'error' not in item, item
            choice = item['choices'][0]
            assert choice['finish_reason'] == 'stop', item
            answer = choice['message']['content']
            usage = item['usage']
    assert expected.lower() in answer.lower(), (label, expected, answer)
    assert usage['completion_tokens'] > 0 and usage['prompt_tokens'] > 0, usage
    result = {'case': label, 'answer': answer, 'seconds': round(time.monotonic()-start, 3),
              'first_content_seconds': first, 'usage': usage}
    print(json.dumps(result), flush=True)
    return result

results.append(check('single', 'What is the capital of France? Answer in one word.', 'Paris'))
results.append(check('stream', 'What is the capital of Germany? Answer in one word.', 'Berlin', True))
questions = [('Italy', 'Rome'), ('Japan', 'Tokyo'), ('Spain', 'Madrid'), ('Canada', 'Ottawa')]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    jobs = [pool.submit(check, 'concurrent-' + country,
                        f'What is the capital of {country}? Answer in one word.', answer)
            for country, answer in questions]
    results.extend(job.result() for job in jobs)
long_prompt = ('The garden has green trees and yellow flowers. ' * 100
               + '\nIgnore the garden description. What is the capital of France? Answer in one word.')
result = check('multi-chunk-prefill', long_prompt, 'Paris')
assert 512 < result['usage']['prompt_tokens'] < 2000, result
results.append(result)
cancel_body = {'model': model, 'messages': [{'role': 'user', 'content': 'Count from one to one hundred.'}],
               'max_tokens': 512, 'temperature': 0, 'stream': True, 'ignore_eos': True}
with request('/v1/chat/completions', cancel_body) as response:
    for line in response:
        if line.startswith(b'data: ') and b'[DONE]' not in line:
            item = json.loads(line[6:])
            if any(c.get('delta', {}).get('content') for c in item.get('choices', [])):
                print('disconnected active stream', flush=True)
                break
    else:
        raise AssertionError('cancellation request produced no content')
results.append(check('reuse-after-disconnect', 'What is two plus two? Answer with one digit.', '4'))
path = pathlib.Path(args.output)
path.write_text(json.dumps({'model': model, 'results': results}, indent=2) + '\n')
print('PASS', len(results), 'requests;', path, flush=True)
