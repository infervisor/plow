#!/usr/bin/env python3
"""Run Coval's paced STT workload against Plow's ASR WebSocket protocol."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import struct
import time
import wave

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer
from websockets.asyncio.client import connect


SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 1_600


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def load_pcm(item, audio_dir):
    path = audio_dir / Path(item["path"]).name
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != item["sha256"]:
        raise ValueError(f"{path}: SHA-256 mismatch")
    with wave.open(str(path), "rb") as source:
        fmt = (source.getframerate(), source.getnchannels(), source.getsampwidth())
        if fmt != (SAMPLE_RATE, 1, 2) or source.getcomptype() != "NONE":
            raise ValueError(f"{path}: expected uncompressed 16 kHz mono PCM16, got {fmt}")
        pcm = source.readframes(source.getnframes())
    duration = float(item["duration_sec"])
    speech_end_ms = item.get("speech_end_offset_ms")
    if isinstance(speech_end_ms, (int, float)):
        trailing_ms = max(0.0, duration * 1000.0 - float(speech_end_ms))
        tail_bytes = round(trailing_ms / 1000.0 * SAMPLE_RATE) * 2
        if 0 < tail_bytes < len(pcm):
            pcm = pcm[:-tail_bytes]
            duration = float(speech_end_ms) / 1000.0
    return pcm, duration


async def transcribe(url, model, language, item, audio_dir, timeout):
    started_total = time.monotonic()
    receiver = None
    row = {
        "kind": "result",
        "id": item["sample_id"],
        "backend": "plow-websocket-paced",
        "reference": item["transcript"],
    }
    try:
        pcm, duration = load_pcm(item, audio_dir)
        row["duration_seconds"] = duration
        async with asyncio.timeout(timeout), connect(
            url, max_size=65_536, close_timeout=1
        ) as socket:
            await socket.send(json.dumps({
                "type": "start",
                "version": 1,
                "model": model,
                "sample_rate": SAMPLE_RATE,
                "format": "pcm_s16le",
                "language": language,
            }))
            ready = json.loads(await socket.recv())
            if ready.get("type") != "ready":
                raise RuntimeError(f"expected ready, received {ready}")
            credit = int(ready["credit_samples"])
            condition = asyncio.Condition()
            partials = []
            error = None
            final = None
            first_partial_at = None
            final_at = None
            audio_start = time.monotonic()

            async def receive():
                nonlocal credit, error, final, first_partial_at, final_at
                async for raw in socket:
                    if isinstance(raw, bytes):
                        continue
                    event = json.loads(raw)
                    kind = event.get("type")
                    now = time.monotonic()
                    if kind == "credit":
                        async with condition:
                            credit += int(event["credit_samples"])
                            condition.notify_all()
                    elif kind == "partial":
                        text = str(event.get("text", "")).strip()
                        if text:
                            partials.append(text)
                            if first_partial_at is None:
                                first_partial_at = now
                    elif kind == "final":
                        final = str(event.get("text", "")).strip()
                        final_at = now
                        if final and first_partial_at is None:
                            first_partial_at = now
                        async with condition:
                            condition.notify_all()
                        return
                    elif kind == "error":
                        error = str(event.get("message", event))
                        async with condition:
                            condition.notify_all()
                        return

            receiver = asyncio.create_task(receive())
            offset = 0
            sequence = 0
            while offset < len(pcm):
                count = min(CHUNK_SAMPLES, (len(pcm) - offset) // 2)
                async with condition:
                    await condition.wait_for(
                        lambda: credit >= count or error is not None or final is not None
                    )
                    if error is not None:
                        raise RuntimeError(error)
                    if final is not None:
                        raise RuntimeError("server finalized before all audio was sent")
                    credit -= count
                chunk = pcm[offset:offset + count * 2]
                await socket.send(struct.pack("<Q", sequence) + chunk)
                offset += len(chunk)
                sequence += 1
                deadline = audio_start + offset / (2 * SAMPLE_RATE)
                delay = deadline - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)

            finalization_start = time.monotonic()
            await socket.send('{"type":"finish"}')
            await receiver
            if error is not None:
                raise RuntimeError(error)
            if final is None or final_at is None:
                raise RuntimeError("stream closed without a final transcript")
            audio_to_final = final_at - audio_start
            row.update({
                "text": final,
                "seconds": audio_to_final,
                "ttft_seconds": None if first_partial_at is None else first_partial_at - audio_start,
                "forced_final_seconds": final_at - finalization_start,
                "ttfs_seconds": max(0.0, audio_to_final - duration),
                "rtf": audio_to_final / duration,
                "partials": partials,
                "total_seconds": time.monotonic() - started_total,
            })
    except Exception as exc:
        row.setdefault("duration_seconds", float(item["duration_sec"]))
        row.update(
            text="",
            error=f"{type(exc).__name__}: {exc}",
            total_seconds=time.monotonic() - started_total,
        )
    finally:
        if receiver is not None:
            if not receiver.done():
                receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
    return row


def summarize(rows):
    normalizer = EnglishTextNormalizer()
    successful = [row for row in rows if "error" not in row]
    per_item = []
    references = []
    hypotheses = []
    for row in rows:
        reference = normalizer(row["reference"])
        hypothesis = normalizer(row["text"])
        result = jiwer.process_words(reference, hypothesis)
        denominator = result.hits + result.substitutions + result.deletions
        per_item.append((result.substitutions + result.deletions + result.insertions) / denominator)
        references.append(reference)
        hypotheses.append(hypothesis)
    corpus = jiwer.process_words(references, hypotheses)

    def metric(name):
        values = [row[name] for row in successful if row.get(name) is not None]
        if not values:
            return None
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }

    return {
        "items": len(rows),
        "failures": len(rows) - len(successful),
        "normalization": "Whisper EnglishTextNormalizer revision 2",
        "whisper_normalizer_version": importlib.metadata.version("whisper-normalizer"),
        "jiwer_version": importlib.metadata.version("jiwer"),
        "corpus_wer": corpus.wer,
        "mean_item_wer": statistics.fmean(per_item),
        "p90_item_wer": percentile(per_item, 0.90),
        "ttft_seconds": metric("ttft_seconds"),
        "forced_final_seconds": metric("forced_final_seconds"),
        "ttfs_seconds": metric("ttfs_seconds"),
        "rtf": metric("rtf"),
    }


async def run(args):
    manifest = json.loads(args.manifest.read_text())
    items = manifest["items"][args.offset:]
    if args.limit is not None:
        items = items[:args.limit]
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(item):
        async with semaphore:
            return await transcribe(
                args.url, args.model, args.language, item, args.audio_dir, args.timeout
            )

    rows = []
    tasks = [asyncio.create_task(bounded(item)) for item in items]
    for task in asyncio.as_completed(tasks):
        row = await task
        rows.append(row)
        print(json.dumps({"id": row["id"], "error": row.get("error"), "done": len(rows)}), flush=True)
    rows.sort(key=lambda row: row["id"])
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    summary = summarize(rows)
    summary.update({
        "dataset": manifest["id"],
        "dataset_version": manifest["version"],
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "url": args.url,
        "model": args.model,
        "concurrency": args.concurrency,
        "chunk_milliseconds": CHUNK_SAMPLES * 1000 / SAMPLE_RATE,
        "result_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    })
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("model")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.timeout <= 0
        or args.offset < 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("concurrency, timeout, and limit must be positive; offset must be nonnegative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
