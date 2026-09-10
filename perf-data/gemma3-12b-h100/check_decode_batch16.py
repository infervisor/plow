import concurrent.futures
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / 'scripts'))
import verify_packed_serve as verify

url = 'http://127.0.0.1:8013'
model = 'checkpoint-fp8'
cases = [(verify.prompt(256 + i, i), 64) for i in range(16)]
expected = [verify.generate(url, model, *case) for case in cases]
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    actual = list(pool.map(lambda case: verify.generate(url, model, *case), cases))
record = {'cases': cases, 'isolated': expected, 'concurrent': actual,
          'mismatches': [i for i in range(16) if expected[i] != actual[i]]}
(Path(__file__).parent / f'decode-batch16-{sys.argv[1]}.json').write_text(json.dumps(record, indent=2))
assert not record['mismatches'], record['mismatches']
print('PASS: 16 concurrent 64-token responses match isolated responses; exact token counts')
