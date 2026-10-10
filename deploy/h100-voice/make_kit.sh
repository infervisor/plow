#!/usr/bin/env bash
# make_kit.sh: assemble the customer H100 kit: ONE runtime (plowrt + its libraries) and model bundles.
#
#   deploy/h100-voice/make_kit.sh <plowrt dir> <audio dir> <out dir> <name>=<bundle dir>...
#
#   <plowrt dir>   the one runtime: plowrt + the cuBLASLt it loads (libcublasLt.so.13) + BUILD.json
#                  [+ plow_verify, the Lean verifier the packets' compiler receipts name]. Every file
#                  in it is shipped and hashed into plowrt/plowrt.sha256.
#   <audio dir>    LibriSpeech dummy clips ls00..ls72.wav + manifest.json (scripts/asr/nvidia/get_audio.py)
#   <out dir>      new kit directory (must not exist)
#   <name>=<dir>   a model bundle served as <name> (silero-vad: silero_vad.pkt + its MANIFEST.json,
#                  SHIP.md, gates.json, recipe.toml)
#   BASELINE=<file> optional single-model baseline copied to BASELINE.md
#
# Layout: models/<name>/ holds the compiled assets only; a bundle's checkpoint/ goes to
# hf/<repo>@<rev>/ (an HF snapshot, named from the bundle's recipe) or hf/<name>@<content id>/ (a
# converted checkpoint), shared by every bundle with identical content. deploy/checkpoints.map pairs
# each model with its hf/ directory (`plow-voice.sh run` passes `--assets DIR,checkpoint=hf/...`).
# Files are hard-linked (same filesystem) or copied. Writes KIT.json, PAIRING.txt and SHA256SUMS.
set -euo pipefail
[ $# -ge 4 ] || { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
rt=$(realpath "$1"); audio=$(realpath "$2"); out=$3; shift 3
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
[ ! -e "$out" ] || { echo "$out exists" >&2; exit 2; }
for f in plowrt BUILD.json; do [ -e "$rt/$f" ] || { echo "$rt/$f missing" >&2; exit 2; }; done
mkdir -p "$out"/{models,hf,plowrt,data/librispeech-dummy,clients/samples}
out=$(realpath "$out")

for f in "$rt"/*; do
  case $(basename "$f") in plowrt.sha256) continue ;; esac
  cp -al "$f" "$out/plowrt/" 2>/dev/null || cp -a "$f" "$out/plowrt/"
done

cp -r "$here/deploy" "$here/docs" "$here/perf" "$out/"
cp -r "$here/eval" "$out/eval"
cp "$here/clients/"*.py "$here/clients/"*.sh "$out/clients/"
cp "$here/README.md" "$here/requirements.txt" "$here/requirements-whisper.txt" "$out/"
[ -n "${BASELINE:-}" ] && cp "$BASELINE" "$out/BASELINE.md"
chmod +x "$out"/deploy/plow-voice.sh "$out"/clients/*.py "$out"/clients/*.sh "$out"/perf/*.py "$out"/eval/*.py

# LibriSpeech dummy validation clips (CC BY 4.0) with a manifest of relative paths.
python3 - "$audio" "$out/data/librispeech-dummy" <<'EOF'
import json, os, shutil, sys
src, dst = sys.argv[1:3]
man = json.load(open(os.path.join(src, "manifest.json")))
for m in man:
    name = os.path.basename(m["path"])
    shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    m["path"] = name
json.dump(man, open(os.path.join(dst, "manifest.json"), "w"), indent=1)
open(os.path.join(dst, "LICENSE.txt"), "w").write(
    "73 utterances of the LibriSpeech ASR corpus (validation 'dummy' split of\n"
    "hf-internal-testing/librispeech_asr_dummy), 16 kHz PCM16 WAV.\n"
    "LibriSpeech (Panayotov et al., 2015, http://www.openslr.org/12) is licensed CC BY 4.0.\n")
EOF
cp "$audio/ls01.wav" "$out/clients/samples/sample_en.wav"

# Bundles -> models/ + hf/, then the pairing (KIT.json, PAIRING.txt) and SHA256SUMS.
python3 - "$out" "$repo" "$@" <<'EOF'
import concurrent.futures as cf, hashlib, json, os, re, shutil, sys, tomllib
out, repo, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
cache = {}

def sha(p):
    st = os.stat(p)
    key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if key not in cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        cache[key] = h.hexdigest()
    return cache[key]

def files(root):
    for d, _, names in os.walk(root):
        for n in names:
            yield os.path.join(d, n)

def hash_all(paths):
    with cf.ThreadPoolExecutor(8) as ex:
        return dict(zip(paths, ex.map(sha, paths)))

def link_tree(src, dst):
    def link(s, d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
    shutil.copytree(src, dst, copy_function=link, symlinks=False)

def hf_name(name, bundle, content_id):
    """hf/<org>--<repo>@<rev12> when the bundle's recipe serves an HF snapshot, else <name>@<id12>."""
    try:
        man = json.load(open(os.path.join(bundle, "MANIFEST.json")))
        recipe = tomllib.load(open(os.path.join(repo, man["recipe"]["path"]), "rb"))
        m = re.fullmatch(r"\{hf:([^@}]+)@([0-9a-f]+)\}", recipe["cell"]["hf_dir"])
        if m:
            return f"{m.group(1).replace('/', '--')}@{m.group(2)[:12]}"
    except (OSError, KeyError, ValueError):
        pass
    return f"{name}@{content_id[:12]}"

checkpoints, by_id = {}, {}
for spec in specs:
    name, bundle = spec.split("=", 1)
    bundle = os.path.realpath(bundle)
    dst = os.path.join(out, "models", name)
    os.mkdir(dst)
    for entry in sorted(os.listdir(bundle)):
        s = os.path.join(bundle, entry)
        if entry == "checkpoint":
            continue
        if os.path.isdir(s):
            link_tree(s, os.path.join(dst, entry))
        else:
            try:
                os.link(s, os.path.join(dst, entry))
            except OSError:
                shutil.copy2(s, os.path.join(dst, entry))
    ck = os.path.join(bundle, "checkpoint")
    if not os.path.isdir(ck):
        continue
    sums = hash_all(sorted(files(ck)))
    content_id = hashlib.sha256("".join(f"{os.path.relpath(p, ck)} {h}\n" for p, h in sums.items()).encode()).hexdigest()
    if content_id not in by_id:
        hf = hf_name(name, bundle, content_id)
        if os.path.exists(os.path.join(out, "hf", hf)):
            hf = f"{hf}-{content_id[:12]}"
        link_tree(ck, os.path.join(out, "hf", hf))
        by_id[content_id] = hf
    checkpoints[name] = {"dir": f"hf/{by_id[content_id]}", "id": content_id}
    print(f"{name}: checkpoint -> hf/{by_id[content_id]}")

with open(os.path.join(out, "deploy", "checkpoints.map"), "w") as f:
    f.write("# <model> <HF checkpoint dir, kit-relative>: written by make_kit.sh, read by plow-voice.sh\n")
    for name, c in sorted(checkpoints.items()):
        f.write(f"{name} {c['dir']}\n")

rt = os.path.join(out, "plowrt")
with open(os.path.join(rt, "plowrt.sha256"), "w") as f:
    for n in sorted(os.listdir(rt)):
        if n not in ("BUILD.json", "plowrt.sha256"):
            f.write(f"{sha(os.path.join(rt, n))}  {n}\n")

plowrt_sha = sha(os.path.join(rt, "plowrt"))
pairs = []
for m in sorted(os.listdir(os.path.join(out, "models"))):
    d = os.path.join(out, "models", m)
    for pkt in sorted(x for x in os.listdir(d) if x.endswith(".pkt")):
        pairs.append([m, f"models/{m}/{pkt}", sha(os.path.join(d, pkt))])
build = json.load(open(os.path.join(rt, "BUILD.json")))
kit = {"kit": os.path.basename(out),
       "plowrt": {"sha256": plowrt_sha, "commit": build.get("plow_commit"), "branch": build.get("branch"),
                  "min_glibc": build.get("min_glibc"), "min_driver": build.get("min_driver"),
                  "cublaslt": build.get("cublaslt")},
       "pairs": pairs, "checkpoints": checkpoints}
json.dump(kit, open(os.path.join(out, "KIT.json"), "w"), indent=1)
with open(os.path.join(out, "PAIRING.txt"), "w") as f:
    f.write("# model, packet, sha256: qualified with plowrt %s (deploy/plow-voice.sh preflight checks these)\n"
            % plowrt_sha)
    for m, path, h in pairs:
        f.write(f"{m} {path} {h}\n")
print(f"KIT.json: {len(pairs)} packets, {len(by_id)} checkpoints, paired with plowrt {plowrt_sha[:12]}")

every = sorted(p for p in files(out) if os.path.basename(p) != "SHA256SUMS")
sums = hash_all(every)
with open(os.path.join(out, "SHA256SUMS"), "w") as f:
    for p in every:
        f.write(f"{sums[p]}  ./{os.path.relpath(p, out)}\n")
print(f"SHA256SUMS: {len(every)} files")
EOF
echo "kit: $out ($(du -sh --apparent-size "$out" | cut -f1))"
