#!/usr/bin/env python3
"""Prepare a deterministic speaker-balanced LibriSpeech ASR evaluation manifest."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import soundfile as sf


def digest(path, kind="sha256"):
    with path.open("rb") as source:
        return hashlib.file_digest(source, kind).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--split", choices=["dev-clean", "dev-other", "test-clean", "test-other"], required=True)
    parser.add_argument("--count", type=int, default=100, help="0 selects the entire split")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 0:
        parser.error("count must be nonnegative")
    expected = dict(line.split()[::-1] for line in args.checksums.read_text().splitlines() if line.strip())
    archive_name = f"{args.split}.tar.gz"
    if expected.get(archive_name) != digest(args.archive, "md5"):
        raise ValueError("archive does not match published split checksum")
    references = {}
    prefix = f"LibriSpeech/{args.split}/"
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if member.isfile() and member.name.startswith(prefix) and member.name.endswith(".trans.txt"):
                for line in archive.extractfile(member).read().decode().splitlines():
                    utterance, text = line.split(" ", 1)
                    if utterance in references:
                        raise ValueError(f"duplicate reference {utterance}")
                    references[utterance] = text
    speakers = {}
    for utterance in references:
        speakers.setdefault(utterance.split("-")[0], []).append(utterance)
    order = lambda value: hashlib.sha256(("plow-asr-v1:" + value).encode()).digest()
    groups = [sorted(speakers[speaker], key=order) for speaker in sorted(speakers, key=order)]
    selected = [group[index] for index in range(max(map(len, groups))) for group in groups if index < len(group)]
    if args.count:
        selected = selected[:args.count]
    wanted = set(selected)
    args.out.mkdir(parents=True, exist_ok=True)
    records = {}
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(prefix) or not member.name.endswith(".flac"):
                continue
            utterance = Path(member.name).stem
            if utterance not in wanted:
                continue
            if utterance in records:
                raise ValueError(f"duplicate audio {utterance}")
            data = archive.extractfile(member).read()
            audio, rate = sf.read(io.BytesIO(data), dtype="int16", always_2d=True)
            record = {"id": utterance, "speaker": utterance.split("-")[0], "split": args.split,
                      "reference": references[utterance], "language": "English",
                      "duration_seconds": len(audio) / rate, "source_sha256": hashlib.sha256(data).hexdigest()}
            if rate != 16000 or audio.shape[1] != 1:
                raise ValueError(f"unexpected LibriSpeech format: {utterance}")
            if not 8000 <= len(audio) <= 480000:
                record["excluded_reason"] = "native duration limit: 0.5–30 seconds"
            else:
                path = args.out / f"{utterance}.wav"
                sf.write(path, audio, rate, subtype="PCM_16")
                record.update(audio=str(path.resolve()), audio_sha256=digest(path))
            records[utterance] = record
    if wanted != records.keys():
        raise ValueError(f"missing audio: {sorted(wanted - records.keys())}")
    manifest = args.out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(records[u], ensure_ascii=False) + "\n" for u in selected))
    metadata = {"dataset": "LibriSpeech", "source": "https://www.openslr.org/12", "license": "CC BY 4.0",
                "split": args.split, "selection": "plow-asr-v1 SHA256 order, round-robin speakers",
                "requested_count": args.count, "selected_count": len(selected), "split_count": len(references),
                "excluded_count": sum("excluded_reason" in r for r in records.values()),
                "archive_sha256": digest(args.archive), "manifest_sha256": digest(manifest),
                "soundfile_version": sf.__version__, "libsndfile_version": sf.__libsndfile_version__}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
