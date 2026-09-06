"""Compare two raw bf16 files (first N elements): usage cmp_two.py A B [n_rows H]"""
import struct, sys

def load(p, n=None):
    b = open(p, 'rb').read()
    m = len(b) // 2 if n is None else min(n, len(b) // 2)
    u = struct.unpack('<%dH' % m, b[:m * 2])
    return [struct.unpack('<f', struct.pack('<I', x << 16))[0] for x in u]

A, B = sys.argv[1], sys.argv[2]
rows, H = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (None, None)
n = rows * H if rows else None
a = load(A, n); b = load(B, n)
m = min(len(a), len(b))
a, b = a[:m], b[:m]
d = [abs(x - y) for x, y in zip(a, b)]
ma = sum(abs(x) for x in a) / m; mb = sum(abs(x) for x in b) / m
big = sum(1 for x, y in zip(a, b) if abs(x - y) > 0.02 * max(1e-2, abs(y)))
print(f"n={m} mean|a|={ma:.4f} mean|b|={mb:.4f} mean|d|={sum(d)/m:.5f} max|d|={max(d):.4f} >2%={100.0*big/m:.2f}%")
if rows:
    for r in (0, rows // 2, rows - 1):
        da = a[r * H:(r + 1) * H]; db = b[r * H:(r + 1) * H]
        dd = sum(abs(x - y) for x, y in zip(da, db)) / H
        print(f"  row {r:3d}: mean|d| {dd:.5f}  a[:4]={[round(v, 4) for v in da[:4]]} b[:4]={[round(v, 4) for v in db[:4]]}")
