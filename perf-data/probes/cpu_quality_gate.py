import argparse
import concurrent.futures
import json
import pathlib
import re
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:18680')
parser.add_argument('--output', required=True)
parser.add_argument('--context', type=int, default=2048)
parser.add_argument('--short-only', action='store_true')
parser.add_argument('--soak-requests', type=int, default=0)
args = parser.parse_args()
base = args.url.rstrip('/')

def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)

model = call('/v1/models')['data'][0]['id']

def wrap(question):
    return ('<bos><|turn>user\n' + question
            + '<turn|>\n<|turn>model\n<|channel>thought\n<channel|>')

def complete(label, question, expected):
    start = time.monotonic()
    result = call('/v1/completions', {
        'model': model, 'prompt': wrap(question), 'add_special_tokens': False,
        'max_tokens': 32, 'temperature': 0, 'return_token_ids': True,
    })
    choice = result['choices'][0]
    text = choice['text'].strip()
    passed = choice['finish_reason'] == 'stop' and bool(
        re.search(r'(?<!\w)' + re.escape(expected) + r'(?!\w)', text, re.I))
    row = {'case': label, 'question': question, 'expected': expected, 'text': text,
           'passed': passed, 'seconds': time.monotonic()-start,
           'usage': result['usage'], 'token_ids': result['token_ids']}
    print(json.dumps({k: v for k, v in row.items() if k not in ('question', 'token_ids')}), flush=True)
    return row

cases = [
    ('france', 'What is the capital of France? Answer in one word.', 'Paris'),
    ('japan', 'What is the capital of Japan? Answer in one word.', 'Tokyo'),
    ('italy', 'What is the capital of Italy? Answer in one word.', 'Rome'),
    ('addition', 'What is 17 plus 25? Reply with digits only.', '42'),
    ('multiply', 'What is 7 times 8? Reply with digits only.', '56'),
    ('chain', 'Start with 9. Multiply by 3, then subtract 5. Reply with the final number only.', '22'),
    ('order', 'Sort 19, 4, and 11 from smallest to largest. Reply with only the middle number.', '11'),
    ('mapping', 'Map A to B, B to C, and C to D. Starting from A, apply the mapping twice. Reply with one letter.', 'C'),
]
results = [complete(*case) for case in cases]

def token_count(prompt):
    return len(call('/tokenize', {'model': model, 'prompt': prompt, 'add_special_tokens': False})['tokens'])

def padded_question(target, code):
    def question(n):
        return (f'Record: the access code is {code}.\n' + 'Routine log entry. ' * n
                + '\nWhat is the access code in the record? Reply with digits only.')
    lo, hi = 0, target
    while lo < hi:
        mid = (lo+hi+1)//2
        if token_count(wrap(question(mid))) <= target:
            lo = mid
        else:
            hi = mid-1
    return question(lo)

rejection = None
if not args.short_only:
    for target, code in [(129, '7319'), (513, '4826'), (1025, '6158'), (args.context-32, '9274')]:
        question = padded_question(target, code)
        row = complete('context-' + str(target), question, code)
        assert target-8 <= row['usage']['prompt_tokens'] <= target, row['usage']
        results.append(row)
    too_long = wrap('Routine log entry. ' * args.context)
    try:
        response = call('/v1/completions', {'model': model, 'prompt': too_long,
                        'add_special_tokens': False, 'max_tokens': 1, 'temperature': 0})
        rejection = {'passed': False, 'response': response}
    except urllib.error.HTTPError as error:
        rejection = {'passed': 400 <= error.code < 500, 'http_status': error.code,
                     'body': error.read().decode()}
    print('context rejection', json.dumps(rejection), flush=True)

soak_start = time.monotonic()
if args.soak_requests:
    def soak(i):
        label, question, expected = cases[i % len(cases)]
        return complete('soak-' + str(i), question + f'\nRequest identifier: {i}.', expected)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results.extend(pool.map(soak, range(args.soak_requests)))
report = {'model': model, 'results': results, 'context_rejection': rejection,
          'soak_requests': args.soak_requests, 'soak_seconds': time.monotonic()-soak_start}
pathlib.Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
assert all(r['passed'] for r in results), 'answer regression; inspect ' + args.output
assert rejection is None or rejection['passed'], 'invalid context was not rejected as a client error'
print('PASS', len(results), 'answers', flush=True)
