import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'scripts'))
import plow_isa

DECODE = 'interp_decode_fp8kv_gq.elf'
PREFILL = 'interp_prefill_fp8kv_mla_moe_gq.elf'
FLASH = 'interp_flash_fp8kv_gq.elf'
NATIVE = {
    'GemmLtPf': ['glm_lt_gfx942.elf'],
    'MoeAiterFp8Pf': ['moe_aiter_adapter_gfx942.elf',
                     'fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flat_pf3.co',
                     'fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co'],
    'IndexTpPf': ['dsa_tp_adapter_gfx942.elf'],
}
SOURCES = ['scripts/build_gfx942.sh', 'runtime/CMakeLists.txt',
           'crates/devgen/src/mla.rs', 'crates/plowrt/src/exec/amd.rs',
           'crates/plowrt/src/exec/amd_tp.rs', 'crates/plowrt/src/exec/amd_gemm_lt.rs',
           'crates/plowrt/src/exec/amd_sparse_mla.rs', 'runtime/amd/op_gemm.h',
           'runtime/amd/op_attention.h', 'runtime/amd/glm_lt_gfx942.json',
           'runtime/amd/glm_lt_decode_gfx942.json']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description='GLM TP8 gfx942 FP8-KV ladder coverage')
    parser.add_argument('--packet-json', type=Path, required=True)
    parser.add_argument('--objects', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--disable-tiers', action='store_true')
    parser.add_argument('--require-specialized', action='store_true')
    args = parser.parse_args()
    raw = args.packet_json.read_text()
    packet = json.loads(raw[raw.index('{'):])
    objects = {}

    def object_info(name):
        if name not in objects:
            path = args.objects / name
            assert path.is_file(), f'missing dispatch object: {path}'
            kernels = plow_isa.notes(path)
            if name == 'glm_lt_gfx942.elf':
                selected = {k['name'] for spec in ('glm_lt_gfx942.json', 'glm_lt_decode_gfx942.json')
                            for k in json.loads((REPO / 'runtime/amd' / spec).read_text())}
                assert selected <= kernels.keys(), 'native GEMM symbols missing'
                kernels = {key: value for key, value in kernels.items() if key in selected}
            objects[name] = dict(sha256=sha(path), globals=plow_isa.globals_u32(path),
                                 kernels=kernels)
        return objects[name]

    def require(name, marker):
        assert object_info(name)['globals'].get(marker), (name, marker)

    tiers = []
    if not args.disable_tiers:
        for path in args.objects.glob('lowrung*'):
            suffix = path.name.removeprefix('lowrung')
            if suffix.isdigit() and int(suffix) > 0 and (path / DECODE).is_file():
                tiers.append((int(suffix), f'{path.name}/{DECODE}'))
        tiers.sort()
    records = []
    for program in packet['programs']:
        counts = Counter(i['op_name'] for i in program['insts'])
        decode = 'FlashMlaDecodeFp8' in counts
        assert decode or 'FlashMlaPrefillFp8' in counts, 'not a supported GLM MLA program'
        rung = program['t']
        primary = next((name for width, name in tiers if rung <= width), DECODE) if decode else PREFILL
        info = object_info(primary)
        require(primary, 'plow_fp8_kv_1')
        record = dict(phase='decode' if decode else 'prefill', rung=rung,
                      interpreter=primary, opcodes=dict(sorted(counts.items())),
                      native_segments={}, projections=[])
        if decode:
            mm = info['globals']['plow_geom_PLOW_GEMV_MM']
            require(primary, f'plow_gemv_mm_cap_{mm}')
            require(primary, 'plow_mla_sparse_fp8_decode_arm')
            if rung > mm:
                require(primary, 'plow_gemv_walk_1')
            expected = min(16, 1 << (rung - 1).bit_length())
            record.update(compiled_gemv_width=mm, matched_width=mm == expected,
                          gemv_passes=(rung + mm - 1) // mm)
            if args.require_specialized:
                assert mm == expected, f'decode rung {rung} uses MM{mm}, expected MM{expected}'
        else:
            if rung >= 2048:
                require(FLASH, 'plow_mla_pf_v2_fp8_arm_1')
                record['attention_interpreter'] = FLASH
            else:
                require(primary, 'plow_cap_d_flash_mla_1')
                record['attention_interpreter'] = primary
            gather = sum(int(i['raw']['fj'][1], 16) != 0 for i in program['insts']
                         if i['op_name'] == 'FlashMlaPrefillFp8')
            if rung < 2048:
                gather = 0
            if gather:
                require('mla_sparse_adapter_gfx942.elf', 'plow_mla_sparse_adapter_fp8_abi_1')
                object_info('mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co')
            record['native_mla_candidates'] = gather
            record['native_mla_condition'] = 'PLOW_MLA_PF_AITER=1; rung >= 2048; selected union; live rows > 0; prior >= 2047. Otherwise the rung attention interpreter.'
        for op, names in NATIVE.items():
            if counts[op]:
                for name in names:
                    object_info(name)
                record['native_segments'][op] = dict(instructions=counts[op], objects=names)
        for (op, shape), count in sorted(Counter(
                (i['op_name'], tuple(i['raw']['i'][:3])) for i in program['insts']
                if i['op_name'].startswith(('Gemv', 'Gemm'))).items()):
            record['projections'].append(dict(op=op, shape=shape, instructions=count))
        records.append(record)
    assert [r['rung'] for r in records if r['phase'] == 'prefill'] == [128, 512, 2048, 8192]
    assert [r['rung'] for r in records if r['phase'] == 'decode'] == [1, 2, 4, 8, 16, 20]
    report = dict(schema='plow.glm-ladder-coverage.v1',
                  scope='Static packet, object-marker and route audit for GLM TP8 gfx942 FP8-KV. Loader pairing and GPU execution are separate checks. Presence does not prove performance optimality.',
                  packet_json_sha256=sha(args.packet_json), disable_tiers=args.disable_tiers,
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  source_sha256={name: sha(REPO / name) for name in SOURCES},
                  checker_sha256=sha(Path(__file__)), programs=records, objects=objects)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {len(records)} ladder programs; {len(objects)} dispatch objects')


if __name__ == '__main__':
    main()
