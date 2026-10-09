# Quality evaluation

`eval/eval.py` measures the served models' output quality against a running server, with the
same metrics and normalization as the release gates, so results compare directly with the
reference values below and in each bundle's `MANIFEST.json` (`gate`).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-whisper.txt   # Whisper only for `tts`
export PLOW_URL=http://127.0.0.1:8000 # PLOW_API_KEY=... when keys are configured
.venv/bin/python eval/eval.py all --out results/eval        # ASR + LLM + TTS; writes eval.json and eval.md
.venv/bin/python eval/eval.py asr                           # one part: asr | tts | llm
.venv/bin/python perf/realtime_vad_test.py --out results/realtime_vad.json   # Realtime server_vad turns
```

Models are discovered from `/v1/models`; `--models a,b` restricts them. The exit status is non-zero
if a request failed or an LLM check failed (quality numbers are reported, not judged).

## What each part does

**ASR (`asr`)**: transcribes every clip of a manifest at concurrency 16 and scores WER after the
gate normalization (lowercase, words of `[a-z0-9']`). Default: the bundled 73 LibriSpeech clips
(`data/librispeech-dummy`, CC BY 4.0). Your own data: `--manifest file.json` (a JSON list of
`{"path", "text", "dur"}`, paths relative to the file), or a LibriSpeech split directory with
`--librispeech <dir>` (e.g. `test-clean` from openslr.org/12; reads the `.flac` and `.trans.txt`
files). `--language-hint` sends `language=en`; by default the model detects the language.

**TTS (`tts`)**: synthesizes each model's gate prompt set (`perf/prompts.py`; chatterbox-mtl: 24
voice-agent sentences, 3 in each of en, hi, zh, ja, es, fr, ar, de; veena: 8 Hindi / English /
code-mixed prompts with their voices; chatterbox: 8 English) as WAV, transcribes the audio with
Whisper large-v3-turbo (forced to the prompt's language), and reports the character error rate
(CER) per prompt, median, mean and per language. Punctuation and symbols are stripped; zh / ja are
compared without spaces. Whisper runs on the GPU beside the server when it has ~3 GB free (the
voice profiles do), else pass `--whisper-device cpu`. Whisper downloads from the Hugging Face hub
on first use; `--whisper /path/to/whisper-large-v3-turbo` uses a local copy. The WAVs stay in
`<out>/tts-<model>/` for listening.

TTS is sampled (temperature > 0, as the models are designed), so CER varies with the seed; the
eval fixes `seed` per prompt. Whisper CER also counts Whisper's own errors (digits, loanwords,
Hindi spelling), which is why the reference itself is not 0 for every language.

**LLM (`llm`)**: functional checks of gemma-4-e4b: six factual answers, greedy determinism, usage
accounting, stop strings, logprobs (top-3, sorted, <= 0), raw completion with `<bos>`, streaming,
the context limit (400 `context_length_exceeded`), and the error envelope for an unknown model.
For model-quality numbers, the release gate compared logits against a reference implementation:
top-1 agreement 0.9845, KL mean 7.4e-4 (`models/gemma-4-e4b/MANIFEST.json`).

**Realtime VAD (`perf/realtime_vad_test.py`)**: Realtime sessions with `server_vad`: three clips
separated by 2 s of silence and 1.5 s of -50 dBFS noise must give three turns, with start / stop
times close to the clip bounds; ten clips alone (each followed by 1.5 s of silence) are scored for
WER. A clip may split into two turns at an in-sentence pause longer than `silence_duration_ms`.

## Reference results (this kit, voice-core profile, one H100)

| check | result | release gate / reference |
|---|---|---|
| qwen3-asr WER, 73 LibriSpeech clips | 3.913% | 3.913% (vLLM reference 3.913%) |
| chatterbox-mtl Whisper CER, 24 prompts | median 0.014, mean 0.050; ar 0.027, de 0.000, en 0.000, es 0.000, fr 0.011, hi 0.108, ja 0.035, zh 0.056 | gate median 0.009 (n=32); reference per language: ar 0.017, de 0, en 0, es 0, fr 0.011, hi 0.108, ja 0.050, zh 0 |
| veena Whisper CER, 8 prompts (voice-veena) | median 0.005, mean 0.121 (the code-mixed Hindi/English prompt scores 0.78 because Whisper answers with an English translation of it; the others <= 0.14) | gate median 0.000 (n=80) |
| gemma-4-e4b functional checks | 14 / 14 | |
| Realtime server_vad | pauses: 3 / 3 turns; 10 clips: WER 4.7% (9 of 10 single-turn) | VAD agent's run: 3 / 3; WER 3.8% on 20 clips |

Other bundles' gate results (each `models/<model>/MANIFEST.json`): qwen3-asr-0.6b WER 4.435%
(c1) / 4.261% (c16); nemotron-3.5-asr 5.13%; chatterbox CER median 0.000; Silero VAD
probabilities within 2.0e-6 of the PyTorch model, decisions at 0.5 identical on 17,321 frames.
