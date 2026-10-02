#!/usr/bin/env python3
"""Render paired llm_grid results from a campaign comparison manifest."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re

import waterfall


def percentile(values):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * .99
    i = int(pos)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (pos - i)


def metrics(runs, gpus):
    duration = sum(d['duration'] for d in runs)
    output = sum(d['total_output_tokens'] for d in runs) / duration
    rates = [d['output_throughput'] for d in runs]
    return dict(
        output_tok_s=output, output_tok_s_gpu=output / gpus,
        total_input_output_tok_s=sum(d['total_input_tokens'] + d['total_output_tokens']
                                    for d in runs) / duration,
        requests_s=sum(d['completed'] for d in runs) / duration,
        ttft_p99_ms=percentile([t * 1000 for d in runs for t in d['ttfts']]),
        tpot_p99_ms=percentile([sum(t) * 1000 / (n - 1) for d in runs
                               for t, n in zip(d['itls'], d['output_lens']) if n > 1]),
        throughput_spread_pct=((max(rates) - min(rates)) / (sum(rates) / len(rates)) * 100
                               if len(rates) > 1 else None),
        peak_gpu_memory_gib=(max(d['_peak_gpu_memory_gib'] for d in runs)
                             if all(d.get('_peak_gpu_memory_gib') is not None for d in runs) else None),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows, sources = [], {}
    for pair in manifest['pairs']:
        roots = [Path(pair['infervisor']), Path(pair['vllm'])]
        groups = [waterfall.grid(str(root)) for root in roots]
        for cell in sorted(set(groups[0]) & set(groups[1]),
                           key=lambda c: (c[0], int(re.search(r'\d+', c)[0]))):
            if not re.fullmatch(r'[gs]\d+', cell):
                continue
            common = []
            for repeat in range(1, manifest['repeats'] + 1):
                tag = f'{cell}.r{repeat}'
                data = [waterfall.bench(str(root), tag) for root in roots]
                if any(d is None for d in data):
                    continue
                for root, d in zip(roots, data):
                    memory_file = root / f'{tag}.peak_gpu_memory_mib.txt'
                    if memory_file.exists() and memory_file.read_text().strip():
                        d['_peak_gpu_memory_gib'] = float(memory_file.read_text()) / 1024
                        sources[str(memory_file)] = hashlib.sha256(memory_file.read_bytes()).hexdigest()
                    assert d['failed'] == 0 and d['completed'] == d['num_prompts'], (root, tag)
                    assert len(d['ttfts']) == len(d['itls']) == len(d['output_lens']) == d['completed']
                    for key in ('num_prompts', 'request_rate', 'max_concurrency',
                                'total_input_tokens', 'total_output_tokens', 'input_lens', 'output_lens'):
                        assert d[key] == data[0][key], (root, tag, key)
                    for key, reconstructed in (
                        ('p99_ttft_ms', percentile([t * 1000 for t in d['ttfts']])),
                        ('p99_tpot_ms', percentile([sum(t) * 1000 / (n - 1)
                                                   for t, n in zip(d['itls'], d['output_lens']) if n > 1]))):
                        assert abs(d[key] - reconstructed) < .01, (root, tag, key)
                    source = next(root.glob(f'{tag}/**/bench.json'))
                    sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
                common.append(data)
            if not common:
                continue
            d = common[0][0]
            row = dict(variant=pair['variant'], cell=cell, input_tokens=d['input_lens'][0],
                       output_tokens=d['output_lens'][0], concurrency=d['max_concurrency'],
                       traffic='greedy' if cell[0] == 'g' else 'sampled T=1 top_p=.95',
                       paired_repeats=len(common), status='complete' if len(common) == manifest['repeats'] else 'provisional',
                       quality=pair['quality'])
            for side, name in enumerate(('infervisor', 'vllm')):
                row.update({f'{name}_{k}': v for k, v in metrics([ds[side] for ds in common], manifest['gpu_count']).items()})
            row['output_throughput_ratio'] = row['infervisor_output_tok_s'] / row['vllm_output_tok_s']
            rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'comparison.json').write_text(json.dumps(dict(manifest=manifest, rows=rows, source_sha256=sources), indent=2) + '\n')
    with (args.out / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    esc = lambda value: html.escape(str(value))
    content = ['<!doctype html><meta charset="utf-8"><title>Gemma 12B FP8 serving comparison</title>',
               '<style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse;margin:20px 0}td,th{border:1px solid #ccc;padding:7px;text-align:right}th{background:#eee}p{max-width:1000px}</style>',
               '<h1>Gemma 12B FP8: Infervisor vs vLLM</h1>']
    for note in manifest['notes']:
        content.append(f'<p>{esc(note)}</p>')
    content.append('<table><tr><th>Configuration</th><th>Value / match status</th></tr>')
    for k, v in manifest['configuration'].items():
        content.append(f'<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>')
    content.append('</table><p>Each metric pair is Infervisor / vLLM. Throughput = total tokens / summed run duration; P99 pools requests across matched repeats. TPOT is request-average time per output token, excluding the first token. Total throughput below means generated output tok/s; CSV also includes input+output tok/s and requests/s. One GPU: throughput/GPU equals total output throughput. Missing memory is not zero. Provisional rows use only the repeats available on both sides.</p>')
    content.append('<table><tr>' + ''.join(f'<th>{x}</th>' for x in ('Variant', 'Input / output', 'Traffic', 'Concurrency', 'Repeats', 'Output tok/s = tok/s/GPU', 'Ratio', 'TTFT P99 ms', 'TPOT P99 ms', 'Peak GPU memory', 'Quality')) + '</tr>')
    for r in rows:
        paired = lambda k: f"{r['infervisor_' + k]:.2f} / {r['vllm_' + k]:.2f}"
        memory = ' / '.join('Not measured' if r[f'{s}_peak_gpu_memory_gib'] is None
                            else f"{r[f'{s}_peak_gpu_memory_gib']:.2f} GiB" for s in ('infervisor', 'vllm'))
        cells = [r['variant'], f"{r['input_tokens']} / {r['output_tokens']}", r['traffic'], r['concurrency'],
                 f"{r['paired_repeats']} ({r['status']})", paired('output_tok_s'), f"{r['output_throughput_ratio']:.3f}",
                 paired('ttft_p99_ms'), paired('tpot_p99_ms'), memory, r['quality']]
        content.append('<tr>' + ''.join(f'<td>{esc(c)}</td>' for c in cells) + '</tr>')
    content.append('</table>')
    (args.out / 'comparison.html').write_text('\n'.join(content) + '\n')
    print(f'{len(rows)} paired cells; {len(sources)} hashed source files; {args.out}')


if __name__ == '__main__':
    main()
