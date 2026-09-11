import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[4]


def commands(folder):
    raw = subprocess.check_output(
        ['ninja', '-C', str(folder), '-t', 'commands', 'gfx950_hsaco'], text=True)
    result = {}
    for line in raw.splitlines():
        words = shlex.split(line)
        runner = next((i for i, s in enumerate(words) if s.endswith('/hipcc_hsaco.sh')), None)
        if runner is None:
            continue
        args = words[runner + 1:]
        output = Path(args[3])
        if not output.name.startswith('interp_decode'):
            continue
        widths = [int(s.split('=')[1]) for s in args if s.startswith('-DPLOW_GEMV_MM=')]
        assert len(widths) == 1, (output, widths)
        normalized = [s for i, s in enumerate(args)
                      if i != 3 and not s.startswith('-DPLOW_GEMV_MM=')]
        result[(output.parent.name, output.name)] = (widths[0], normalized)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = []
    cases = [(1, [], []), (4, [], [1, 2]), (20, [], [1, 2, 4, 8]),
             (20, ['-DPLOW_HSACO_DECODE_TIERS=OFF'], []),
             (20, ['-DPLOW_GEMV_MM=8'], [1, 2, 4])]
    with tempfile.TemporaryDirectory(prefix='plow-ladder-build-') as temp:
        for number, (batch, extra, expected) in enumerate(cases):
            folder = Path(temp) / str(number)
            cmd = ['cmake', '-S', str(REPO / 'runtime'), '-B', str(folder), '-G', 'Ninja',
                   '-DPLOW_GFX950_HSACO=ON', '-DPLOW_HSACO_ARCH=gfx942',
                   f'-DPLOW_DECODE_BATCH={batch}', '-DPLOW_GEMV_WALK=ON', *extra]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            built = commands(folder)
            primary = {name: data for (parent, name), data in built.items()
                       if not parent.startswith('lowrung')}
            assert primary
            tiers = {(parent, name): data for (parent, name), data in built.items()
                     if parent.startswith('lowrung')}
            assert set(tiers) == {(f'lowrung{w}', name) for w in expected for name in primary}
            for (parent, name), (width, flags) in tiers.items():
                assert width == int(parent.removeprefix('lowrung'))
                assert flags == primary[name][1], (parent, name, 'non-width flags drifted')
            assert 'interp_decode_fp8kv_gq.elf' in primary
            records.append(dict(batch=batch, options=extra, widths=expected,
                                primary_variants=sorted(primary), tier_objects=len(tiers)))
    args.out.write_text(json.dumps(records, indent=2) + '\n')
    print(f'PASS: {len(records)} CMake configurations; all precision/scheduler tier flags match')


if __name__ == '__main__':
    main()
