# Qwen3-ASR results summary (H100)

Numbers are carried from the cited records. No baseline (vLLM or other) ASR serving result is
recorded in the repo, and there is no strict `campaign.py report` for ASR, so nothing here is a
qualified plow-vs-baseline win.

- Model: Qwen/Qwen3-ASR-1.7B `7278e1e7`, BF16, one H100 SXM, TP1.
- Recipe: [`recipes/infervisor/qwen3-asr/sm90a-h100-tp1.toml`](../../../../recipes/infervisor/qwen3-asr/sm90a-h100-tp1.toml).
  Build and gate steps: [h100-speech-recipes.md](../../../runtime/h100-speech-recipes.md).
  Mechanisms and the c1 latency breakdown: [asr.md](../../../runtime/asr.md#h100-single-request-latency-qwen3-asr-17b-sm90a-tp1).
- Latest reproduction: main `fc0271e8` (plowrt sha256 `be62ab42…`), packet `561172a6`, encoder
  `0eb16493`, 2026-10-03, 2 repeats per cell.
- Gate `asr_wer` PASS: WER 3.913% on the LibriSpeech-clean subset, 0 errors (unchanged in every arm).

## Performance

Client: `scripts/asr/nvidia/served_bench.py`. Two manifests are in use, and RTFx depends on
which one:
- `manifest_x4.json`: 292 clips, about 2.5 s of wall time per cell.
- The 73-clip, 481 s manifest from `scripts/asr/nvidia/get_audio.py`.

| Metric | Recorded | main `fc0271e8` | pre-squash `dd1be445` |
|---|---:|---:|---:|
| c1 latency p50 (73 clips) | 55.2 ms | 54.5 ms [54.5-54.6] | |
| RTFx c64 / c128, `manifest_x4` (292 clips) | 728 / 758 | 739 / 775 | 731 / 766 |
| RTFx c64 / c128, 73 clips | | 645 / 588 | |
| RTFx c16, 73 clips | | 523 | |

The apparent -11% / -22% RTFx regression in the first main run was a manifest mismatch: the
recorded 728 / 758 came from `manifest_x4.json`, not the 73-clip set (bisect 2026-10-04). The
73-clip c64/c128 cells last about 0.8 s, so they are noise-sensitive.

## Evidence

- `/opt/dlami/nvme/lava-tts/repro-main/`: `res/` (bench cells), `gates/qwen3-asr/`,
  `report/gates-score.log`, `report/summ.py`.
- `/opt/dlami/nvme/lava-tts/bisect/` (2026-10-04 regression bisect, ASR and E4B).
