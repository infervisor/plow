"""Portable raw-sample integrity checks, not benchmark authenticity or FP correctness."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ("ctrl", "treat", "ctrl2", "treat2")


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
    if not lengths or len(set(lengths)) != 1 or type(lengths[0]) is not int or lengths[0] <= 0:
        raise ValueError("heterogeneous or invalid input lengths")
    concurrency = client["max_concurrency"]
    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError("invalid concurrency")
    ttft = [x * 1e3 for x in client["ttfts"]]
    intervals = client["itls"]
    if len(ttft) != len(lengths) or len(intervals) != len(lengths) or any(not row for row in intervals):
        raise ValueError("incomplete request samples")
    tpot = [sum(row) / len(row) * 1e3 for row in intervals]
    for xs in (ttft, *intervals, tpot):
        stats(xs)
    if client.get("completed") != len(lengths):
        raise ValueError("benchmark contains incomplete requests")
    return (lengths[0], concurrency), {"ttft_ms": ttft, "tpot_ms": tpot}


def capture(runs):
    if set(runs) != set(ARMS) or len({Path(p).resolve() for p in runs.values()}) != 4:
        raise ValueError("four distinct benchmark runs required")
    return {"schema": 1, "scope": "campaign_sample_integrity", "arms": {
        arm: {"record": artifact(Path(runs[arm]) / "run-record.json"),
              "clients": [artifact(path) for path in sorted((Path(runs[arm]) / "client").glob("in*_c*.json"))]}
        for arm in ARMS}}


def validate(evidence, request):
    if evidence.get("schema") != 1 or evidence.get("scope") != "campaign_sample_integrity" \
            or set(evidence.get("arms", {})) != set(ARMS):
        raise ValueError("unsupported campaign evidence")
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
            cell, metrics = samples(client)
            if len(client["input_lens"]) != int(record["protocol"]["NPROMPT"]) \
                    or client.get("output_lens") != [int(record["protocol"]["OUTLEN"])] * len(client["input_lens"]):
                raise ValueError("client differs from declared request/output-length protocol")
            if cell in seen:
                raise ValueError("duplicate client cell")
            seen.add(cell)
            for metric, xs in metrics.items():
                population[(arm, *cell, metric)] = (client_artifact["sha256"], xs)
        if not seen:
            raise ValueError("empty arm")
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
        if key[0].startswith("treat"):
            for relation, arm in (("control_of", "ctrl"), ("repeat_control_of", "ctrl2"),
                                  ("repeat_treatment_of", "treat2" if key[0] == "treat" else "treat")):
                other = entries.get(entry.get(relation), {})
                source = other.get("sample_source", {})
                if (source.get("arm"), source.get("input_len"), source.get("concurrency"), other.get("metric")) != (arm, *key[1:]):
                    raise ValueError("ledger arm relation differs from raw run provenance")
    if used != set(population):
        raise ValueError("ledger omits benchmark cells")
