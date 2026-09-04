import re, sys
s = re.sub(r'\x1b\[[0-9;]*m', '', open(sys.argv[1]).read())
ps = [float(x) for x in re.findall(r'"p50": ([0-9.]+)', s)]
ck = re.findall(r'"output_checksum": "([^"]+)"', s)
err = re.findall(r'Error: .{0,100}', s)
ttft = [x for x in ps if 500 < x < 3000]
tpot = [x for x in ps if 15 < x < 60]
e2e = [x for x in ps if x > 4000]
print(f"ttft={ttft[:1]} tpot={tpot[:1]} e2e={e2e[:1]} ck={ck[:1]} {err[:1]}")
