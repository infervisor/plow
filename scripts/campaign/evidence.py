"""Portable raw-sample integrity checks, not benchmark authenticity or FP correctness."""
import hashlib
import json
import math
from pathlib import Path
import statistics

try:
    from .client_latency import export_identity
except ImportError:
    from client_latency import export_identity

ARMS = ("ctrl", "treat", "ctrl2", "treat2")
METRICS = ("ttft_ms", "tpot_ms")
SAMPLE_SEMANTICS = "stream_intervals_per_output_token_reconstructed_v1"
EXACT_SAMPLE_SEMANTICS = "vllm029_request_latency_minus_ttft_per_output_token_v1"


def has_exact_export(client):
    return any(field in client for field in ("request_latencies", "request_success", "client_latency_export"))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def artifact(path):
    raw = Path(path).read_bytes()
    return {"sha256": digest(raw), "text": raw.decode("utf-8")}


def document(item):
    if set(item) != {"sha256", "text"} or digest(item["text"].encode()) != item["sha256"]:
        raise ValueError("evidence artifact hash mismatch")
    return json.loads(item["text"])


def stats(xs):
    if not xs or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in xs):
        raise ValueError("empty, nonfinite or nonpositive timing samples")
    median = statistics.median(xs)
    return {"median": median, "mad": statistics.median(abs(x - median) for x in xs),
            "n": len(xs), "p95": sorted(xs)[min(len(xs) - 1, int(0.95 * len(xs)))]}


def samples(client):
    lengths = client["input_lens"]
    if not lengths or any(type(n) is not int or n <= 0 for n in lengths) or len(set(lengths)) != 1:
        raise ValueError("heterogeneous or invalid input lengths")
    concurrency = client["max_concurrency"]
    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError("invalid concurrency")
    raw_ttft = client["ttfts"]
    stats(raw_ttft)
    ttft = [x * 1e3 for x in raw_ttft]
    intervals = client["itls"]
    outputs = client["output_lens"]
    if len(ttft) != len(lengths) or len(intervals) != len(lengths) or len(outputs) != len(lengths):
        raise ValueError("incomplete request samples")
    if any(type(n) is not int or n < 0 for n in outputs) or any(
            not isinstance(row, list) or any(type(x) not in (int, float)
                or not math.isfinite(x) or x < 0 for x in row) for row in intervals):
        raise ValueError("invalid output counts or streaming intervals")
    if has_exact_export(client):
        latencies = client.get("request_latencies")
        successes = client.get("request_success")
        if client.get("client_latency_export") != export_identity() \
                or not isinstance(latencies, list) or len(latencies) != len(lengths) \
                or not isinstance(successes, list) or len(successes) != len(lengths) \
                or any(success is not True for success in successes) \
                or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in latencies):
            raise ValueError("incomplete, failed or unpinned exact client latency export")
        tpot = [(latency - first) / (n - 1) * 1e3
                for latency, first, n in zip(latencies, raw_ttft, outputs) if n > 1]
    else:
        # Ordinary detailed JSON omits latency. Reported aggregates stay in the
        # raw artifact; this reconstruction is not the hidden latency arithmetic.
        tpot = [sum(row) / (n - 1) * 1e3 for row, n in zip(intervals, outputs) if n > 1]
    if type(client.get("completed")) is not int or client["completed"] != len(lengths):
        raise ValueError("benchmark contains incomplete requests")
    return (lengths[0], concurrency), {"ttft_ms": ttft, "tpot_ms": tpot}


def declared_integers(protocol, field):
    value = protocol.get(field)
    if not isinstance(value, str) or not value.split():
        raise ValueError(f"missing declared {field}")
    words = value.split()
    if any(not word.isascii() or not word.isdecimal() for word in words):
        raise ValueError(f"malformed declared {field}")
    values = [int(word) for word in words]
    if any(n <= 0 for n in values) or len(set(values)) != len(values):
        raise ValueError(f"nonpositive or duplicate declared {field}")
    return values


def declared_scalar(protocol, field):
    values = declared_integers(protocol, field)
    if len(values) != 1:
        raise ValueError(f"expected one declared {field}")
    return values[0]


def capture(runs):
    if set(runs) != set(ARMS) or len({Path(p).resolve() for p in runs.values()}) != 4:
        raise ValueError("four distinct benchmark runs required")
    bundle = {"schema": 2, "scope": "campaign_sample_integrity",
            "sample_semantics": SAMPLE_SEMANTICS, "required_metrics": list(METRICS), "arms": {
        arm: {"record": artifact(Path(runs[arm]) / "run-record.json"),
              "clients": [artifact(path) for path in sorted((Path(runs[arm]) / "client").glob("in*_c*.json"))]}
        for arm in ARMS}}
    exact = {has_exact_export(document(client)) for arm in bundle["arms"].values()
             for client in arm["clients"]}
    if len(exact) > 1:
        raise ValueError("cannot mix reconstructed and exact client samples")
    if exact == {True}:
        bundle.update(schema=3, sample_semantics=EXACT_SAMPLE_SEMANTICS,
                      client_identity=export_identity())
    return bundle


def check_reported_aggregates(client, metrics):
    for metric, xs in metrics.items():
        stats(xs)
        # The client aggregates seconds with NumPy, then converts to ms. Bound
        # summation/conversion rounding, not unobserved timestamp differences.
        tolerance = 8 * len(xs) * math.ulp(max(xs))
        for name, expected in (("mean", statistics.fmean(xs)), ("median", statistics.median(xs))):
            reported = client.get(f"{name}_{metric}")
            if type(reported) not in (int, float) or not math.isfinite(reported) \
                    or not math.isclose(reported, expected, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError("exact samples differ from authoritative client aggregates")


def validate(evidence, request):
    if evidence.get("schema") == 1:
        raise ValueError("historical mean-ITL evidence has no declared-scope qualification; regenerate evidence")
    exact = evidence.get("schema") == 3
    if evidence.get("schema") not in (2, 3) or evidence.get("scope") != "campaign_sample_integrity" \
            or evidence.get("sample_semantics") != (EXACT_SAMPLE_SEMANTICS if exact else SAMPLE_SEMANTICS) \
            or evidence.get("required_metrics") != list(METRICS) \
            or set(evidence.get("arms", {})) != set(ARMS):
        raise ValueError("unsupported campaign evidence")
    if exact and evidence.get("client_identity") != export_identity():
        raise ValueError("exact evidence lacks pinned client identity")
    records, population = {}, {}
    record_hashes = set()
    for arm in ARMS:
        item = evidence["arms"][arm]
        record = records[arm] = document(item["record"])
        record_hashes.add(item["record"]["sha256"])
        if record.get("gate") is not True or record.get("contended") is not False \
                or type(record.get("bench_rc")) is not int or record["bench_rc"] != 0:
            raise ValueError("failed, contended or incomplete benchmark arm")
        if not record.get("cell") or not record.get("gpu") or not record.get("protocol"):
            raise ValueError("missing workload/hardware/protocol provenance")
        protocol = record["protocol"]
        declared_cells = {(length, concurrency) for length in declared_integers(protocol, "IN_LENS")
                          for concurrency in declared_integers(protocol, "CONCS")}
        nprompt = declared_scalar(protocol, "NPROMPT")
        outlen = declared_scalar(protocol, "OUTLEN")
        if outlen <= 1:
            raise ValueError("TTFT/TPOT campaign qualification requires output length > 1")
        if not is_digest(record.get("hashes", {}).get("model.pkt")):
            raise ValueError("missing full packet identity")
        execution = record.get("execution_artifacts", {})
        if record.get("execution_artifacts_unchanged") is not True \
                or execution.get("assets", {}).get("model.pkt") != record["hashes"]["model.pkt"]:
            raise ValueError("missing or changed observed execution artifacts")
        for field in ("runtime_sha256", "recipe_sha256", "runtime_environment_sha256", "serve_args_sha256"):
            value = execution.get(field, "")
            if not is_digest(value):
                raise ValueError("incomplete execution artifact fingerprint")
        for group in ("assets", "objects"):
            if not isinstance(execution.get(group), dict) or any(
                    not name or Path(name).name != name or not is_digest(value)
                    for name, value in execution[group].items()):
                raise ValueError("invalid observed object/asset fingerprint")
        seen = set()
        for client_artifact in item["clients"]:
            client = document(client_artifact)
            if has_exact_export(client) != exact:
                raise ValueError("client sample semantics differ from declared evidence scope")
            cell, metrics = samples(client)
            if exact:
                check_reported_aggregates(client, metrics)
            if len(client["input_lens"]) != nprompt \
                    or client.get("output_lens") != [outlen] * len(client["input_lens"]):
                raise ValueError("client differs from declared request/output-length protocol")
            if cell in seen:
                raise ValueError("duplicate client cell")
            seen.add(cell)
            for metric, xs in metrics.items():
                population[(arm, *cell, metric)] = (client_artifact["sha256"], xs)
        if seen != declared_cells:
            raise ValueError("observed cells differ from declared IN_LENS x CONCS")
        if arm == "ctrl":
            expected_cells = seen
        elif seen != expected_cells:
            raise ValueError("four-arm workload coverage differs")
    if len(record_hashes) != 4:
        raise ValueError("reused benchmark run record")
    for arm in ARMS[1:]:
        for field in ("cell", "gpu", "protocol"):
            if records[arm][field] != records["ctrl"][field]:
                raise ValueError(f"four-arm {field} mismatch")
    for left, right in (("ctrl", "ctrl2"), ("treat", "treat2")):
        for field in ("hashes", "serve_env", "overrides", "execution_artifacts"):
            if records[left].get(field) != records[right].get(field):
                raise ValueError(f"repeat arm {field} mismatch")
    used = set()
    prefixes = set()
    entries = {entry["id"]: entry for entry in request["ledger"]}
    if len(entries) != len(request["ledger"]):
        raise ValueError("duplicate ledger identity")
    for entry in request["ledger"]:
        source = entry.get("sample_source", {})
        key = (source.get("arm"), source.get("input_len"), source.get("concurrency"), entry["metric"])
        if key in used or key not in population:
            raise ValueError("ledger source missing or duplicated")
        used.add(key)
        sha, xs = population[key]
        if source.get("client_sha256") != sha or entry.get("samples") != xs or entry.get("stats") != stats(xs):
            raise ValueError("ledger differs from raw client samples")
        if entry.get("recipe_digest") != records[key[0]]["hashes"]["model.pkt"]:
            raise ValueError("ledger packet identity mismatch")
        rung = entry.get("rung", {})
        suffix = f"/serve/in{key[1]}-c{key[2]}-out{outlen}"
        label = rung.get("digest")
        if not isinstance(label, str) or not label.endswith(suffix) or len(label) <= len(suffix) \
                or rung.get("role") != "serve" or type(rung.get("rows")) is not int \
                or rung["rows"] != key[1] or rung.get("topology") != f"C{key[2]}" \
                or entry.get("better") != "lower":
            raise ValueError("ledger rung or metric direction differs from raw client cell")
        prefixes.add(label[:-len(suffix)])
        if key[0].startswith("treat"):
            for relation, arm in (("control_of", "ctrl"), ("repeat_control_of", "ctrl2"),
                                  ("repeat_treatment_of", "treat2" if key[0] == "treat" else "treat")):
                other = entries.get(entry.get(relation), {})
                source = other.get("sample_source", {})
                if (source.get("arm"), source.get("input_len"), source.get("concurrency"), other.get("metric")) != (arm, *key[1:]):
                    raise ValueError("ledger arm relation differs from raw run provenance")
    if used != set(population):
        raise ValueError("ledger omits benchmark cells")
    if len(prefixes) != 1:
        raise ValueError("ledger uses different campaign rung prefixes")
    expected_touched = {entry["id"] for entry in entries.values()
                        if entry["sample_source"]["arm"] == "treat"}
    expected_serving = {entry["id"] for entry in entries.values()
                        if entry["sample_source"]["arm"] in ("treat", "treat2")}
    touched = request.get("touched", [])
    touched_ids = [item.get("treat") for item in touched]
    serving = request.get("serving", [])
    if set(touched_ids) != expected_touched or len(touched_ids) != len(expected_touched) \
            or set(serving) != expected_serving or len(serving) != len(expected_serving) \
            or request.get("tier4") is not True or request.get("untouched") != []:
        raise ValueError("certificate scope omits or changes measured cells/metrics")
    if any(item.get("rung") != entries[item["treat"]]["rung"]["digest"] for item in touched):
        raise ValueError("certificate rung differs from raw client cell")
