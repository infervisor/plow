import array, math
a, b = "/tmp/g4c/l2r/ref/e2b.L4.c16384", "/tmp/g4c/l2r/ref/e2b.L4.c16384.chk"
def rd(p, t): x = array.array(t); x.frombytes(open(p, "rb").read()); return x
for n in ("x_in.f32", "ref.attn.f32", "ref.out.f32"):
    x, y = rd(f"{a}/{n}", "f"), rd(f"{b}/{n}", "f")
    print(n, "rel_rms %.2e" % math.sqrt(sum((p - q) ** 2 for p, q in zip(x, y)) / sum(p * p for p in x)))
x, y = rd(f"{a}/kcache.bf16", "H"), rd(f"{b}/kcache.bf16", "H")
print("kcache identical frac %.4f" % (sum(p == q for p, q in zip(x, y)) / len(x)))
