import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    source = json.loads((root / 'serving-source.json').read_text())
    builds = source['source_build']['builds']
    assert sha(Path('/tmp/tp-glm53-row-norm/plowrt')) == source['runtime_sha256']
    assert sha(Path('/tmp/tp-glm53-row-norm/assets-on/model.pkt')) == source['packet_sha256']
    for arm in ('safe', 'swz'):
        for name, digest in source['arms'][arm].items():
            assert sha(root / f'serving-{arm}' / name) == digest, (arm, name)
    differing = [name for name in source['arms']['safe']
                 if source['arms']['safe'][name] != source['arms']['swz'][name]]
    assert differing == ['interp_flash_fp8kv_gq.elf'], differing
    for arm, build in builds.items():
        frozen = root / 'objects' / arm / 'source'
        for name, digest in build['source_sha256'].items():
            assert sha(frozen / name) == digest, (arm, name)
        assert sha(root / 'objects' / arm / 'interp_flash_fp8kv_gq.elf') == build['sha256']

    numerical = {}
    library_builds = json.loads((root / 'safe/build-record.json').read_text())
    for defer in (0, 1):
        data = json.loads((root / f'safe/defer{defer}-numerics.json').read_text())
        assert len(data['cases']) == 36
        assert all(c['exact_schedule_match'] and c['relative_l2'] < .01 for c in data['cases'])
        for arm, value in (('off', 0), ('on', 1)):
            digest = library_builds[f'defer{defer}-swz{value}']['sha256']
            assert sha(root / f'safe/defer{defer}-swz{value}.so') == digest
            assert data['library_sha256'][arm] == digest
        assert data['checker_sha256'] == sha(Path(__file__).with_name('check.py'))
        numerical[f'defer{defer}'] = data
    timing = json.loads((root / 'safe/timing.json').read_text())
    assert len(timing['cases']) == 12
    assert timing['library_sha256'] == numerical['defer1']['library_sha256']

    metrics = {}
    quality = {}
    lengths = None
    keys = ['duration', 'completed', 'failed', 'total_input_tokens', 'total_output_tokens',
            'output_throughput', 'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms',
            'mean_tpot_ms', 'median_tpot_ms', 'p99_tpot_ms',
            'mean_itl_ms', 'median_itl_ms', 'p99_itl_ms']
    for arm in ('safe', 'swz', 'safe-repeat'):
        data = json.loads((root / f'serve-{arm}.json').read_text())
        assert data['completed'] == data['num_prompts'] == 20 and data['failed'] == 0
        assert data['max_concurrency'] == 20 and not any(data['errors'])
        current = {k: data[k] for k in ('input_lens', 'output_lens')}
        if lengths is None:
            lengths = current
        assert lengths == current, arm
        assert sum(data['input_lens']) == data['total_input_tokens']
        assert sum(data['output_lens']) == data['total_output_tokens']
        metrics[arm] = {k: data[k] for k in keys}
        quality[arm] = json.loads((root / f'quality-{arm}.json').read_text())
        cells = quality[arm]['cells']
        assert len(cells) == 18 and all(c['passed'] for c in cells), arm
    baseline = (metrics['safe']['output_throughput'] + metrics['safe-repeat']['output_throughput']) / 2
    result = dict(
        schema='plow.mla-probability-layout.v1', artifact_root=str(root),
        source=source, library_builds=library_builds, numerical=numerical, timing=timing,
        metrics=metrics, quality=quality, lengths=lengths,
        swizzle_throughput_change_percent=100 * (metrics['swz']['output_throughput'] / baseline - 1),
        baseline_repeat_change_percent=100 * (metrics['safe-repeat']['output_throughput'] /
                                              metrics['safe']['output_throughput'] - 1),
        sha256={p.name: sha(p) for p in sorted(root.glob('*.json'))
                if p.name.startswith(('serve-', 'quality-'))},
        recorder_sha256=sha(Path(__file__)),
        limitations=[
            'One ordered safe/swizzle/safe sequence, 20 prompts per run; no full100 or H200 parity claim.',
            'Both serving arms fix inactive dense FP8 tail-scale reads; this does not isolate the guard cost.',
            'Standalone normalized FP32 oracle samples query rows; guards and layout equality cover all outputs.',
            'Standalone timing excludes union construction, quantization, merge, interpreter and TP.',
            'Library and full interpreter builds use the recorded frozen sources, not every change at current HEAD.',
        ])
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('metrics', 'swizzle_throughput_change_percent',
                                          'baseline_repeat_change_percent')}, indent=2))


if __name__ == '__main__':
    main()
