"""Pinned, post-timing vLLM client exports; never modifies the installed image."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

IMAGE = "vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1"
CLIENT_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py"
SOURCE_SHA256 = "ca017dceb15805283d2c45549f3c449cf7b1d411c34fb0d6f41eecf456b64e94"
ENDPOINT_SHA256 = "2e873b7c67075c371370fe80555de9e289050054b0758f6ba28645d0f2108e11"
EXPORT_SCHEMA = "plow_vllm029_post_timing_latencies_v1"
OVERLAY_SHA256 = "21871046e55e1dc1f30ffa856100e86e44a5dcabdb0b42641226f19400c0ce42"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source_identity():
    return {"schema": EXPORT_SCHEMA, "image": IMAGE, "source_sha256": SOURCE_SHA256,
            "endpoint_sha256": ENDPOINT_SHA256}


def export_identity():
    return {**source_identity(), "overlay_sha256": OVERLAY_SHA256}


def overlay_source(source, endpoint_sha):
    if digest(source.encode()) != SOURCE_SHA256 or endpoint_sha != ENDPOINT_SHA256:
        raise ValueError("pinned vLLM client source identity changed")
    anchor = '            "output_lens": actual_output_lens,\n'
    if source.count(anchor) != 1:
        raise ValueError("client latency export anchor is not unique")
    fields = (
        '            "request_latencies": [output.latency for output in outputs],\n'
        '            "request_success": [output.success for output in outputs],\n'
        f'            "client_latency_export": {{**{source_identity()!r}, '
        '"overlay_sha256": __import__("hashlib").sha256('
        '__import__("pathlib").Path(__file__).read_bytes()).hexdigest()},\n'
    )
    modified = source.replace(anchor, anchor + fields)
    tree = ast.parse(modified)
    benchmark = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                     and node.name == "benchmark")
    duration = [node for node in ast.walk(benchmark) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "benchmark_duration"
                        for target in node.targets)]
    export = [node for node in ast.walk(benchmark) if isinstance(node, ast.Dict)
              and any(isinstance(key, ast.Constant) and key.value == "request_latencies"
                      for key in node.keys)]
    if len(duration) != 1 or len(export) != 1 or export[0].lineno <= duration[0].lineno:
        raise ValueError("client export does not follow the timed benchmark")
    if digest(modified.encode()) != OVERLAY_SHA256:
        raise ValueError("pinned client overlay implementation changed")
    return modified


def fetch_source():
    code = (
        "import pathlib,hashlib,json; "
        f"p=pathlib.Path({CLIENT_PATH!r}); "
        "e=p.parent/'lib/endpoint_request_func.py'; "
        "print(json.dumps({'source':p.read_text(),"
        "'endpoint_sha256':hashlib.sha256(e.read_bytes()).hexdigest()}))"
    )
    result = subprocess.run(["sudo", "-n", "docker", "run", "--rm", "--network", "none",
                             "--entrypoint", "python3", IMAGE, "-c", code],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def write_overlay(directory, source, endpoint_sha):
    image = overlay_source(source, endpoint_sha).encode()
    metadata = {**source_identity(), "overlay_sha256": digest(image)}
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in (("serve.py", image),
                       ("identity.json", (json.dumps(metadata, sort_keys=True) + "\n").encode())):
        path = directory / name
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"refusing to overwrite changed client overlay: {path}")
        else:
            with path.open("xb") as stream:
                stream.write(data)
        path.chmod(0o444)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    original = fetch_source()
    metadata = write_overlay(args.output_dir, original["source"], original["endpoint_sha256"])
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
