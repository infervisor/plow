#!/usr/bin/env bash
# make_kit.sh: assemble the customer voice kit from a frozen speech release and a plowrt build.
#
#   deploy/h100-voice/make_kit.sh <release dir> <plowrt dir> <audio dir> <out dir> [vad dir]
#
#   <release dir>  plow-h100-speech-<sha>/ (bundles under <model>/sm90a-h100-tp1/)
#   <plowrt dir>   plowrt + libcublasLt.so.12 + BUILD.json + plowrt.sha256 [+ plow_verify, the Lean
#                  verifier the packets' compiler receipts name; plowrt re-checks them at load]
#   <audio dir>    LibriSpeech dummy clips ls00..ls72.wav + manifest.json (scripts/asr/nvidia/get_audio.py)
#   <out dir>      new kit directory (must not exist)
#   [vad dir]      Silero VAD bundle (silero_vad.pkt, MANIFEST.json, SHIP.md, gates.json, recipe.toml),
#                  shipped as models/silero-vad
#
# Bundles are hard-linked (same filesystem) or copied; orpheus goes under experimental/. Writes
# KIT.json (the plowrt <-> packet pairing the preflight checks) and SHA256SUMS.
set -euo pipefail
rel=$(realpath "$1"); rt=$(realpath "$2"); audio=$(realpath "$3"); out=$4; vad=${5:-}
here=$(cd "$(dirname "$0")" && pwd)
[ ! -e "$out" ] || { echo "$out exists" >&2; exit 2; }
mkdir -p "$out"/{models,experimental,plowrt,data/librispeech-dummy,clients/samples}
out=$(realpath "$out")
link() { cp -al "$1" "$2" 2>/dev/null || cp -a "$1" "$2"; }

for f in plowrt libcublasLt.so.12 BUILD.json plowrt.sha256 plow_verify; do
  [ -e "$rt/$f" ] || [ "$f" = plow_verify ] || { echo "$rt/$f missing" >&2; exit 2; }
  [ -e "$rt/$f" ] && link "$rt/$f" "$out/plowrt/$f"
done
for m in qwen3-asr qwen3-asr-0.6b nemotron-3.5-asr veena chatterbox chatterbox-mtl gemma-4-e4b; do
  link "$rel/$m/sm90a-h100-tp1" "$out/models/$m"
done
link "$rel/orpheus/sm90a-h100-tp1" "$out/experimental/orpheus"
if [ -n "$vad" ]; then
  mkdir "$out/models/silero-vad"
  for f in silero_vad.pkt MANIFEST.json SHIP.md gates.json recipe.toml commit; do link "$vad/$f" "$out/models/silero-vad/$f"; done
fi
cp "$rel/BASELINE.md" "$out/BASELINE.md"

cp -r "$here/deploy" "$here/docs" "$here/perf" "$out/"
cp -r "$here/eval" "$out/eval"
cp "$here/clients/"*.py "$here/clients/"*.sh "$out/clients/"
cp "$here/README.md" "$here/requirements.txt" "$here/requirements-whisper.txt" "$out/"
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

# Pairing: the plowrt and the packets it was qualified with (preflight checks these hashes).
python3 - "$out" "$rel" <<'EOF'
import hashlib, json, os, sys
out, rel = sys.argv[1:3]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
pairs = []
for base in ("models", "experimental"):
    for m in sorted(os.listdir(os.path.join(out, base))):
        d = os.path.join(out, base, m)
        for pkt in sorted(x for x in os.listdir(d) if x.endswith(".pkt")):
            pairs.append([m, f"{base}/{m}/{pkt}", sha(os.path.join(d, pkt))])
build = json.load(open(os.path.join(out, "plowrt", "BUILD.json")))
kit = {"kit": os.path.basename(out), "release": os.path.basename(rel),
       "plowrt": {"sha256": sha(os.path.join(out, "plowrt", "plowrt")), "commit": build.get("plow_commit"),
                  "min_glibc": build.get("min_glibc")},
       "pairs": pairs}
json.dump(kit, open(os.path.join(out, "KIT.json"), "w"), indent=1)
with open(os.path.join(out, "PAIRING.txt"), "w") as f:
    f.write("# model, packet, sha256: qualified with plowrt %s (deploy/plow-voice.sh preflight checks these)\n"
            % kit["plowrt"]["sha256"])
    for m, path, h in pairs:
        f.write(f"{m} {path} {h}\n")
print(f"KIT.json: {len(pairs)} packets paired with plowrt {kit['plowrt']['sha256'][:12]}")
EOF
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "kit: $out ($(du -sh --apparent-size "$out" | cut -f1)); SHA256SUMS: $(wc -l < "$out/SHA256SUMS") files"
