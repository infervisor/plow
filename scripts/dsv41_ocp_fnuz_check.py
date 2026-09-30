"""Is reinterpreting OCP e4m3 bits as FNUZ e4m3 exactly a halving?

OCP  e4m3fn : value = (-1)^s * 2^(e-7) * (1 + m/8),  e in 1..15 (e=15,m=7 = NaN)
FNUZ e4m3   : value = (-1)^s * 2^(e-8) * (1 + m/8),  e in 1..15
denormals   : OCP 2^-6 * m/8 ; FNUZ 2^-7 * m/8

Same layout, bias differs by one => same bits read as FNUZ is exactly half.
If that holds, an OCP->FNUZ move costs nothing as long as the block scale is
doubled, and a ue8m0 scale doubles by adding 1 to its exponent byte.
"""
import torch

ocp = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
fnuz = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fnuz).float()

exact = 0
mismatch = []
for i in range(256):
    o, f = ocp[i].item(), fnuz[i].item()
    if o != o or f != f:            # NaN on either side
        mismatch.append((i, o, f, "nan"))
        continue
    if f == o / 2.0:
        exact += 1
    else:
        mismatch.append((i, o, f, "value"))

print(f"bit patterns where FNUZ == OCP/2 exactly : {exact}/256")
print(f"non-matching patterns                    : {len(mismatch)}")
for i, o, f, why in mismatch:
    print(f"   0x{i:02x}  ocp={o!r:>12}  fnuz={f!r:>12}  ({why})")

# ue8m0 scale: adding 1 to the exponent byte doubles the scale, exactly.
s = torch.tensor([120, 127, 128, 200], dtype=torch.uint8)
before = s.view(torch.float8_e8m0fnu).float()
after = (s + 1).view(torch.float8_e8m0fnu).float()
print()
print("ue8m0 scale byte +1 doubles the scale:")
for b, a in zip(before.tolist(), after.tolist()):
    print(f"   {b:>12g} -> {a:>12g}   exact={a == b * 2}")
