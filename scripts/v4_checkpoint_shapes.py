import json, struct, glob, collections
d = '/workspace/models/DeepSeek-V4-Flash-0731'
tot = collections.Counter()
tens = {}
for f in sorted(glob.glob(d + '/model-*.safetensors')):
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k == '__metadata__':
            continue
        tens[k] = (v['dtype'], v['shape'])
        tot[v['dtype']] += 1
print("dtypes seen:", dict(tot))
json.dump(tens, open('/tmp/v4_tensors.json', 'w'))
print("n tensors", len(tens))
for pref in ['model.layers.0.', 'model.layers.2.', 'model.layers.3.']:
    print("=== ", pref)
    for k in sorted(tens):
        if k.startswith(pref) and ('experts.' not in k or 'shared' in k):
            print("  ", k, tens[k])
