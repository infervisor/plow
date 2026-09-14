import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'scripts'))
import plow_isa

READ128 = [list(range(b + a, b + a + 4)) + list(range(b + c, b + c + 4))
           for b in (0, 32) for a, c in ((0, 20), (4, 16), (8, 28), (12, 24))]
WRITE128 = [list(range(b, b + 8)) for b in range(0, 64, 8)]
SOURCES = [
    'runtime/amd/op_gemm.h', 'runtime/amd/op_moe.h', 'runtime/amd/op_attention.h',
    'runtime/amd/amd_common.h', 'runtime/amd/interp.hip',
    'runtime/amd/dsa_tp_adapter.hip', 'runtime/amd/mla_sparse_adapter.hip',
    'runtime/amd/moe_aiter_adapter.hip', 'runtime/amd/mla_materialized_opus.hip',
    'runtime/amd/glm_lt_gfx942.json', 'runtime/amd/glm_lt_decode_gfx942.json',
    'runtime/nvidia/op_gemm.cuh', 'runtime/nvidia/op_gemm_sm90.cuh',
    'runtime/nvidia/op_gemm_splitk.cuh', 'runtime/nvidia/op_attention.cuh',
    'runtime/nvidia/op_attention_sm90.cuh', 'runtime/nvidia/op_mla.cuh',
    'crates/plowrt/src/exec/amd_gemm_lt.rs',
    'crates/plowrt/src/exec/amd_sparse_mla.rs',
    'runtime/CMakeLists.txt', 'scripts/build_gfx942.sh',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conflicts(address, phases):
    worst, example = 0, None
    for phase in phases:
        banks = defaultdict(set)
        for lane in phase:
            start = address(lane)
            assert start % 16 == 0
            for byte in range(0, 16, 4):
                word = (start + byte) // 4
                banks[word % 32].add(word)
        n = max(map(len, banks.values()))
        if n > worst:
            worst = n
            example = dict(lanes=phase, byte_addresses=[address(l) for l in phase])
    return dict(max_distinct_dwords_per_bank_per_phase=worst, example=example)


def address_cases():
    cases = {}

    def add(name, address, expected, phases=READ128):
        result = conflicts(address, phases)
        assert result['max_distinct_dwords_per_bank_per_phase'] == expected, name
        cases[name] = result

    assert sorted(l for p in READ128 for l in p) == list(range(64))
    assert sorted(l for p in WRITE128 for l in p) == list(range(64))
    # Known controls distinguish true aliasing from a same-address read broadcast.
    assert conflicts(lambda l: 0, READ128)['max_distinct_dwords_per_bank_per_phase'] == 1
    assert conflicts(lambda l: l * 128, READ128)['max_distinct_dwords_per_bank_per_phase'] == 8
    for bk in (32, 64, 128):
        for step in range(0, bk, 16):
            add(f'gemm_read_bk{bk}_k{step}', lambda l: 2 * (
                (l % 32) * bk + ((step + 8 * (l // 32)) ^ (((l % 32) & (bk // 8 - 1)) * 8))),
                2 if bk == 32 else 1)
        add(f'gemm_write_bk{bk}', lambda l: 2 * (
            (l * 8 // bk) * bk + ((l * 8 % bk) ^ (((l * 8 // bk) & (bk // 8 - 1)) * 8))),
            1, WRITE128)
    # Candidate changes are address models only; production code is unchanged.
    for step in (0, 16):
        add(f'gemm_bk32_candidate_k{step}', lambda l: 2 * (
            (l % 32) * 32 + ((step + 8 * (l // 32)) ^ ((((l % 32) >> 1) & 3) * 8))), 1)
    add('gemm_bk32_candidate_write', lambda l: 2 * (
        (l // 4) * 32 + ((l % 4 * 8) ^ (((l // 4 >> 1) & 3) * 8))), 1, WRITE128)
    for stride in (72, 136, 520, 584):
        add(f'mfma32_padded_stride{stride}', lambda l: 2 * ((l % 32) * stride + 8 * (l // 32)), 1)
    for swz in (0, 16):
        for nt in (0, 1):
            for kt in range(18):
                add(f'mla_v2_k_swz{swz}_nt{nt}_kt{kt}', lambda l: 2 * (
                    (nt * 16 + l % 16) * 584 + swz * ((nt * 16 + l % 16) // 8)
                    + kt * 32 + 8 * (l // 16)), 2)
    add('mla_v2_p', lambda l: 2 * ((l % 16) * 32 + 8 * (l // 16)), 2)
    add('mla_v2_p_candidate', lambda l: 2 * (
        (l % 16) * 32 + ((8 * (l // 16)) ^ (((l % 16) % 4) * 8))), 1)
    for row in range(32):
        for layout in (lambda k: k ^ (((row >> 1) & 3) * 8),
                       lambda k: k ^ ((row & 3) * 8)):
            assert sorted(layout(k) for k in range(32)) == list(range(32))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--packet-json', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = dict(schema='plow.lds-address-review.v1',
                  scope='gfx942 ds_read_b128/ds_write_b128 address model; no GPU performance measurement',
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  source_sha256={p: sha(REPO / p) for p in SOURCES},
                  checker_sha256=sha(Path(__file__)), address_cases=address_cases())
    if args.packet_json:
        raw = args.packet_json.read_text()
        packet = json.loads(raw[raw.index('{'):])
        report['packet_json_sha256'] = sha(args.packet_json)
        report['packet_opcode_inventory'] = [dict(t=p['t'], opcodes=dict(Counter(
            i['op_name'] for i in p['insts']))) for p in packet['programs']]
    if args.assets:
        names = ('interp_flash_fp8kv_gq.elf', 'interp_decode_fp8kv_gq.elf',
                 'glm_lt_gfx942.elf', 'mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co',
                 'fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flat_pf3.co',
                 'fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co')
        report['objects'] = {}
        for name in names:
            path = args.assets / name
            selected = None
            if name == 'glm_lt_gfx942.elf':
                selected = set()
                for spec in ('glm_lt_gfx942.json', 'glm_lt_decode_gfx942.json'):
                    selected.update(p['name'] for p in json.loads((REPO / 'runtime/amd' / spec).read_text()))
                asm = subprocess.check_output([plow_isa.tool('llvm-objdump'), '-d', '--mcpu=gfx942',
                    '--disassemble-symbols=' + ','.join(sorted(selected)), str(path)], text=True)
                functions = plow_isa.parse(asm)
            else:
                functions = plow_isa.counts(path)
            total = Counter()
            for f in functions:
                total.update(f.insn)
            notes = plow_isa.notes(path)
            if selected:
                notes = {k: v for k, v in notes.items() if k in selected}
                assert {f.name for f in functions} == selected
            report['objects'][name] = dict(sha256=sha(path), notes=notes,
                selected_symbols=sorted(selected) if selected else None,
                geometry={k: v for k, v in plow_isa.globals_u32(path).items()
                          if k.startswith('plow_geom_')},
                static_instruction_counts=dict(total),
                mla_functions=[dict(name=f.name, instructions=dict(f.insn))
                               for f in functions if 'd_flash_mla_prefill_v2' in f.name])
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f"PASS: {len(report['address_cases'])} address cases; report {args.out}")


if __name__ == '__main__':
    main()
