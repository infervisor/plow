"""Export the LibriSpeech dummy validation split as 16 kHz PCM16 WAVs plus a manifest."""
import json
import sys
from pathlib import Path

import io

import datasets
import soundfile as sf

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/asr-work/audio")
out.mkdir(parents=True, exist_ok=True)
ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", datasets.Audio(decode=False))
man = []
for i, r in enumerate(ds):
    wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
    p = out / f"ls{i:02d}.wav"
    sf.write(p, wav, sr, subtype="PCM_16")
    man.append({"path": str(p), "text": r["text"], "dur": len(wav) / sr})
(out / "manifest.json").write_text(json.dumps(man, indent=1))
print(len(man), "clips", round(sum(m["dur"] for m in man), 2), "s")
