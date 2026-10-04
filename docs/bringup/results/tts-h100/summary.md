# TTS results summary (H100): Veena, Chatterbox, Chatterbox Multilingual

Numbers are carried from the cited records. There is no strict `campaign.py report` for speech
(`serving_comparison.py` covers LLM serving), so no TTS row is a qualified plow-vs-baseline win;
the only same-client baseline recorded is vLLM on Veena's token stream.

| Model | Recipe | Packet (main `fc0271e8` repro) | Gates (all PASS) |
|---|---|---|---|
| Veena | [`recipes/infervisor/veena/sm90a-h100-tp1.toml`](../../../../recipes/infervisor/veena/sm90a-h100-tp1.toml) | `c45cf845`, codec `c7026b0f` | `tts_cer`: Whisper CER median 0.005, n=80 |
| Chatterbox | [`recipes/infervisor/chatterbox/sm90a-h100-tp1.toml`](../../../../recipes/infervisor/chatterbox/sm90a-h100-tp1.toml) | `bfc34919`, s3gen `64ef9339` | `tts_cer` 0 (n=32); `s3gen_rel_l2` mel 1.23e-5 |
| Chatterbox MTL | [`recipes/infervisor/chatterbox-mtl/sm90a-h100-tp1.toml`](../../../../recipes/infervisor/chatterbox-mtl/sm90a-h100-tp1.toml) | `62abc913`, s3gen `9720eec7` | `tts_cer` 0 (n=32), worst language fr 0.148; `s3gen_rel_l2` mel 1.23e-5 |

Reproduction: main `fc0271e8` (plowrt sha256 `be62ab42…`), CUDA 12.9, one H100 80GB HBM3 SXM,
one lease 2026-10-03 22:01-22:57 UTC, 2 repeats per cell, 0 failed requests. Build and gate
steps: [h100-speech-recipes.md](../../../runtime/h100-speech-recipes.md). Mechanisms and the full
optimization log: [tts.md](../../../runtime/tts.md).

## Performance

Mean of 2 repeats, [min-max]. "Recorded" is the value the recipes were qualified with
(h100-speech-recipes.md before this consolidation).

| Model | Metric | Recorded | main `fc0271e8` | Baseline |
|---|---|---:|---:|---:|
| Veena | out tok/s c64 (ISL 128 / OSL 512, greedy) | ~13,780 | 13,824 [13,819-13,830] | vLLM 12,863 |
| Veena | out tok/s c128 | ~20,590 | 20,653 [20,621-20,685] | vLLM 19,972 |
| Veena | stream aps c64; TTFA p50 | 76.2; 138 ms | 77.4; 136.6 ms | |
| Chatterbox | stream aps c64 | 58.4 | 61.9 [61.8-62.0] | |
| Chatterbox MTL | stream steady aps c200, 0 failed | 71.7 | 78.8 [78.7-78.9]* (plain aps 77.6) | stock fp32, 1 request: 1.31 aps |

\* Steady aps on main is an approximation (audio delivered in the middle 60% of the wall time,
scratch `tts_bench_steady.py`); the original definition of the recorded 71.7 is unknown. Plain
`audio_s_per_s` also exceeds it.

Other cells on main: Veena stream c1 3.4 aps (TTFA 56 ms), c8 21.3, c32 52.5, c128 100.3 (TTFA
220 ms). Chatterbox stream c1 8.9 aps (TTFA 132 ms), c8 31.0. MTL c1 8.7 aps (TTFA 133 ms), c16
46.4, c200 TTFA p50 6.2 s. Single runs vary about ±1.5%.

The Veena vLLM numbers come from the same `llm_grid.sh` client on the token stream
([tts.md](../../../runtime/tts.md#veena)); there is no recorded vLLM+SNAC audio-stream result.

## Evidence

- `/opt/dlami/nvme/lava-tts/repro-main/`: `res/` (bench cells), `gates/<model>/` (gate captures),
  `report/gates-score.log`, `report/summ.py` (means and spread), `src/scripts/tts/tts_bench_steady.py`.
- Bench commands: `$S/job.sh` there (gate `run.sh`, then `plowrt serve` with the recipe env and
  `tts_bench` arms).
