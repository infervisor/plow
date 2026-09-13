# Frozen Gemma 4 31B H100 release

Location: `/opt/dlami/nvme/plow-releases/gemma4-31b-h100-20260909`.
Each profile contains regular-file copies of `bin/plowrt`, `assets/`, weights,
tokenizer, linked libraries, a checksum manifest and sample workloads.

Release index SHA-256:
`3577f42048a285c15907fd16d184399753fe0d072aeba8d6aa3ea7d573187685`.
All payload checksums passed; 58 release-local sample requests passed. The
payload is read-only and occupies approximately 203 GiB.

| Profile | Weights / KV | Physical slots | Decode rungs | Unified execution |
|---|---|---:|---|---|
| `bf16-unified` (default) | BF16 / BF16 | 8 | 1, 2, 4, 8 | Functionally validated |
| `fp8-unified` | FP8 / BF16 | 16 | 1, 2, 4, 8, 16 | Functionally validated |
| `bf16-fp8kv` | BF16 / FP8 | 16 | 1, 2, 4, 8, 16 | Ordinary fallback |

All profiles retain prefill 128/512/1024 and a 32768-token context limit, including
output. FP8 weights use the validated mixed-activation profile. The unified
profiles pin the tested runtime corresponding to `3ca64e9`; the FP8-KV profile
preserves the earlier `25fb3d7` checkpoint and its existing qualification.

```sh
cd /opt/dlami/nvme/plow-releases/gemma4-31b-h100-20260909
python3 verify.py
./serve.sh

# Choose a different profile; run only one on this H100 at a time.
PLOW_PROFILE=fp8-unified PORT=8080 ./serve.sh
PLOW_PROFILE=bf16-fp8kv PORT=8080 ./serve.sh

# Run workloads from another terminal.
python3 bf16-unified/workloads.py --concurrency 8 --max-tokens 128

# Disable unified dispatch while retaining prefix caching.
./serve.sh --token-batch=false

# Direct CLI access using the bundled ELF loader and libraries.
./plowrt.sh --help
```

Use each profile's `assets/` and `bin/plowrt` together. `plowrt.sh` selects the
profile using `PLOW_PROFILE`, like `serve.sh`. The bundled loader avoids reliance
on the build machine's Nix store. The host still supplies the NVIDIA driver;
`PLOW_LIBCUDA` overrides its path.

These are frozen inputs for testing production traffic. Functional serving,
cache reuse and matching-computation logits have been checked; sustained SLO
and application-quality qualification remain open. Experimental tensor-core
decode is not enabled. The sample SLO is 600000 ms; set `SLO_MS` to the latency
budget being tested. `MAX_QUEUED_REQUESTS` defaults to 64. The API binds on the
runtime's default interface; use the deployment's controlled ingress.

The release stays fixed while development continues. `release.json` records
profile manifest hashes. Logs and qualification scope are in `evidence/` and
each profile's manifest. Storage is local to this instance; preserve the entire
directory when moving it to another compatible H100 host.
