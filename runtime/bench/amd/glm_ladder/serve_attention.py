#!/usr/bin/env python3
"""Check exact append lengths after shared prefixes, including one-row tails."""

import argparse
import json
import re
from pathlib import Path
import urllib.request

RUNGS = [1, 2, 4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('output', type=Path)
    parser.add_argument('--mode', choices=['retrieval', 'exact'], default='retrieval')
    parser.add_argument('--contexts', type=int, nargs='+', default=[1024, 16384, 65536])
    parser.add_argument('--rungs', type=int, nargs='+', default=RUNGS)
    args = parser.parse_args()
    if any(x not in RUNGS for x in args.rungs) or any(x < 1024 for x in args.contexts):
        parser.error('use supported append rungs and contexts >= 1024')
    model = json.load(urllib.request.urlopen(args.url + '/v1/models'))['data'][0]['id']

    def post(endpoint, body):
        request = urllib.request.Request(args.url + endpoint, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    filler = post('/tokenize', dict(model=model, prompt=
        'The observatory records the weather each morning. Keep these notes for later reference.\n',
        add_special_tokens=False))['tokens']

    def complete(prompt, output=1):
        result = post('/v1/completions', dict(model=model, prompt=prompt,
            max_tokens=output, temperature=0, ignore_eos=output == 1,
            add_special_tokens=False, return_token_ids=True))
        assert result['usage']['prompt_tokens'] == len(prompt), result['usage']
        assert 1 <= result['usage']['completion_tokens'] <= output, result['usage']
        assert result['token_ids']['prompt'] == prompt
        assert len(result['token_ids']['completion']) == result['usage']['completion_tokens']
        return result

    records = []
    for context in args.contexts:
        prefix = (filler * (context // len(filler) + 1))[:context]
        if args.mode == 'exact':
            complete(prefix + [1000])
        for ix, rows in enumerate(args.rungs):
            code = str(1000 + (context * 17 + rows * 31) % 9000)
            if args.mode == 'retrieval':
                header = post('/tokenize', dict(model=model, prompt=
                    f'The archive lookup key for this record is {code}.\n',
                    add_special_tokens=False))['tokens']
                trailer = post('/tokenize', dict(model=model, prompt=
                    '\nReturn only the archive lookup key from the first line.\nKey:',
                    add_special_tokens=False))['tokens']
                fill = context + rows - len(header) - len(trailer)
                assert fill >= 0
                prompt = header + (filler * (fill // len(filler) + 1))[:fill] + trailer
                complete(prompt[:context] + [0 if prompt[context] != 0 else 1])
                output = 16
            else:
                suffix = [1001 + ix] + (filler * (rows // len(filler) + 1))[:rows - 1]
                prompt = prefix + suffix
                output = 1
            first = complete(prompt, output)
            cached = first['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
            repeat = complete(prompt, output)
            texts = [r['choices'][0]['text'].strip() for r in [first, repeat]]
            record = dict(context=context, append_rows=rows, cached_tokens=cached,
                          first=first['token_ids']['completion'],
                          repeat=repeat['token_ids']['completion'], text=texts)
            if args.mode == 'retrieval':
                record['expected'] = code
                record['correct'] = all(re.match(re.escape(code) + r'(?:\D|$)', text)
                                        is not None for text in texts)
            else:
                record['correct'] = record['first'] == record['repeat']
            record['correct'] &= cached == context
            records.append(record)
            args.output.write_text(json.dumps({'mode': args.mode, 'cases': records}, indent=2) + '\n')
            assert record['correct'], record
            print(f'PASS: context={context} append={rows} mode={args.mode} output={texts}', flush=True)


if __name__ == '__main__':
    main()
