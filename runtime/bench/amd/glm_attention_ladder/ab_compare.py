import csv, sys
def load(p):
    rows = [r for r in csv.DictReader(l for l in open(p) if l.count(',') >= 10)]
    return {(r['op'], r['rows'], r['ctx'], r['arm'], r['ns'], r['gf'], r['blocks']): r for r in rows}
a, b = load(sys.argv[1]), load(sys.argv[2])
la, lb = sys.argv[3] if len(sys.argv) > 3 else 'A', sys.argv[4] if len(sys.argv) > 4 else 'B'
print(f"op,rows,ctx,arm,ns,gf,{la}_us,{lb}_us,delta,{la}_maxabs,{lb}_maxabs,{lb}_note")
for k in sorted(a, key=lambda k: (k[0], int(k[1]), int(k[2]), k[3], int(k[4]))):
    if k not in b: continue
    ua, ub = float(a[k]['median_us']), float(b[k]['median_us'])
    print(f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},{k[5]},{ua:.1f},{ub:.1f},{(ub/ua-1)*100:+.1f}%,{a[k]['max_abs']},{b[k]['max_abs']},{b[k]['note']}")
