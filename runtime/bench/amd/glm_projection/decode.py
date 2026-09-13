import argparse, ctypes, hashlib, json, statistics, re, struct
from pathlib import Path
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['library', 'direct'], required=True)
parser.add_argument('--library', type=Path, required=True)
parser.add_argument('--capture', type=Path, required=True)
parser.add_argument('--object', type=Path)
parser.add_argument('--selected', type=Path)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--export', type=Path)
parser.add_argument('--rows', type=int, nargs='+', default=[16, 20])
parser.add_argument('--arms', nargs='+', choices=['plow', 'hipblas', 'hipblaslt'])
args = parser.parse_args()
assert all(1 <= rows <= 20 for rows in args.rows), 'capture contains 20 rows'
assert args.mode == 'library' or args.arms is None
capture = args.capture
if args.export:
    args.export.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
torch.cuda.set_device(0)
assert 'gfx942' in torch.cuda.get_device_properties(0).gcnArchName
lib = ctypes.CDLL(str(args.library.resolve()))
fn = lib.plow_projection
fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint] * 3 + [ctypes.c_void_p]
fn.restype = ctypes.c_int
width_fn = getattr(lib, 'plow_projection_width', None)
if width_fn is not None:
    width_fn.argtypes = []
    width_fn.restype = ctypes.c_uint
compiled_width = width_fn() if width_fn is not None else None
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if args.mode == 'direct':
    assert args.selected and args.object
    configs = json.loads(args.selected.read_text())
    obj = args.object
    assert sha(obj) == 'efa5b0365bedc2effa52265c85eded14fb63febd9c067bab37138d99db607db5'
    hip = ctypes.CDLL('libamdhip64.so')
    hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    hip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    module = ctypes.c_void_p()
    assert hip.hipModuleLoad(ctypes.byref(module), str(obj).encode()) == 0
shapes = [('q_a', 'x', 'wqa', 2048, 6144), ('kv_latent', 'x', 'wkv', 512, 6144), ('q_absorb', 'qlat', 'wabs', 4096, 2048), ('o_proj', 'oat', 'wout', 6144, 2048), ('router', 'xn2', 'wrouter', 256, 6144), ('shared_down', 'shfu', 'wshared', 6144, 256)]
flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device='cuda')
records = []

def read(name):
    return torch.frombuffer(bytearray((capture / (name + '.b001.bin')).read_bytes()), dtype=torch.bfloat16).clone()

def timing(call):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    warm = torch.cuda.CUDAGraph()
    with torch.cuda.graph(warm):
        for _ in range(20):
            call()
    cold = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cold):
        call()
    result = {}
    for mode, graph, reps in [('warm', warm, 20), ('cold', cold, 1)]:
        samples = []
        for _ in range(11):
            if mode == 'cold':
                flush.fill_(17)
            s, e = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            s.record()
            graph.replay()
            e.record()
            e.synchronize()
            samples.append(s.elapsed_time(e) * 1000 / reps)
        result[mode + '_us'] = statistics.median(samples)
        result[mode + '_samples_us'] = samples
    return result
for name, xfile, wfile, n, k in shapes:
    wh = read(wfile).reshape(n, k)
    xh = read(xfile)[:20 * k].reshape(20, k)
    assert torch.isfinite(wh).all() and torch.isfinite(xh).all()
    oracle = xh.double() @ wh.double().T
    w = wh.cuda()
    for rows in args.rows:
        x = xh[:rows].cuda()
        for arm in (args.arms or ['plow', 'hipblas', 'hipblaslt']) if args.mode == 'library' else ['direct-linear', 'direct-xcc']:
            storage = torch.full((rows * n + 256,), 27499, device='cuda', dtype=torch.int16)
            out = storage[128:-128].view(torch.bfloat16).reshape(rows, n)
            kernel_info = {}
            if arm.startswith('direct-'):
                config = next((d for d in configs if (int(d['M']), int(d['N']), int(d['K'])) == (n, rows, k)))
                symbol = config['kernel_name']
                mt0, mt1, _ = map(int, re.search('MT(\\d+)x(\\d+)x(\\d+)', symbol).groups())
                wg = re.search('_WG(\\d+)_(\\d+)_(\\d+)$', symbol)
                threads = int(wg[1]) * int(wg[2]) * int(wg[3])
                kernel = ctypes.c_void_p()
                assert hip.hipModuleGetFunction(ctypes.byref(kernel), module, symbol.encode()) == 0
                mapping = re.search('_WGM(\\d+)_WGMXCC(\\d+)_WGMXCCGn1$', config['solution_Name'])
                info1 = 1 if arm == 'direct-linear' else int(mapping[1]) | int(mapping[2]) << 16
                grid = (n + mt0 - 1) // mt0 * ((rows + mt1 - 1) // mt1)
                raw = bytearray(160)
                struct.pack_into('<8I', raw, 0, 1, 1, info1, grid, n, rows, 1, k)
                struct.pack_into('<4Q', raw, 32, out.data_ptr(), out.data_ptr(), w.data_ptr(), x.data_ptr())
                struct.pack_into('<8I', raw, 64, n, n * rows, n, n * rows, k, k * n, k, k * rows)
                struct.pack_into('<2f', raw, 96, 1.0, 0.0)
                struct.pack_into('<Q', raw, 140, out.data_ptr())
                buffer = ctypes.create_string_buffer(bytes(raw))
                size = ctypes.c_size_t(160)
                extra = (ctypes.c_void_p * 5)(1, ctypes.addressof(buffer), 2, ctypes.addressof(size), 3)

                def call():
                    rc = hip.hipModuleLaunchKernel(kernel, grid, 1, 1, threads, 1, 1, 0, torch.cuda.current_stream().cuda_stream, None, extra)
                    assert rc == 0, rc
                kernel_info = dict(solution_index=config['solution_index'], kernel_name=symbol, info1=info1, grid=grid, threads=threads)
            elif arm == 'plow':

                def call():
                    rc = fn(out.data_ptr(), x.data_ptr(), w.data_ptr(), rows, n, k, torch.cuda.current_stream().cuda_stream)
                    assert rc == 0, rc
            else:
                torch.backends.cuda.preferred_blas_library(arm)

                def call():
                    torch.mm(x, w.T, out=out)
            out.fill_(float('nan'))
            call()
            torch.cuda.synchronize()
            actual = out.cpu().double()
            assert torch.isfinite(actual).all()
            relative = float(torch.linalg.vector_norm(actual - oracle[:rows]) / torch.linalg.vector_norm(oracle[:rows]))
            assert relative < 0.01, (name, rows, arm, relative)
            assert (storage[:128] == 27499).all() and (storage[-128:] == 27499).all()
            if args.export and arm == 'direct-xcc':
                (args.export / f'{name}-{rows}.bin').write_bytes(out.view(torch.uint8).cpu().numpy().tobytes())
            expected = out.clone()
            for _ in range(3):
                out.fill_(float('nan'))
                call()
                torch.cuda.synchronize()
                assert torch.equal(out, expected)
            record = dict(shape=name, rows=rows, n=n, k=k, arm=arm, compiled_width=compiled_width if arm == 'plow' else None, relative_l2=relative, guards=True, poison_reuses=3, **kernel_info, **timing(call))
            records.append(record)
            print(json.dumps(record), flush=True)
            args.out.with_suffix('.partial.json').write_text(json.dumps(records, indent=2) + '\n')
report = dict(records=records, source_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('decode_kernels.hip'), args.library]}, capture_sha256={name + '.b001.bin': sha(capture / (name + '.b001.bin')) for _, x, w, _, _ in shapes for name in [x, w]}, torch=torch.__version__, hip=torch.version.hip, gpu=torch.cuda.get_device_name(0), mode=args.mode, object_sha256=sha(args.object) if args.object else None, selected_sha256=sha(args.selected) if args.selected else None, scope='Captured layer77 rank0 BF16 projections, full CPU FP64 oracle. Direct mode uses raw HIP module dispatch, inline160B ABI GSU1. Library mode uses PyTorch BLAS preferences. Graph timings exclude interpreter and TP; not HSA runtime or model qualification.')
args.out.write_text(json.dumps(report, indent=2) + '\n')
