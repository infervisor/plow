import hashlib
import json
import struct
import subprocess
import sys


SCRIPT = __file__.replace("tests/test_mla_boundary_abi.py", "mla_boundary_abi.py")


def payload(tmp_path, name, words):
    path = tmp_path / name
    path.write_bytes(struct.pack(f"<{len(words)}H", *words))
    return str(path)


def test_seal_and_pack_exact_materialized_qkv(tmp_path):
    tokens = tmp_path / "tokens.u32le"
    tokens.write_bytes(struct.pack("<2I", 4, 9))
    t = {
        "latent.q": ([2, 1], [1, 2]),
        "latent.kv": ([2, 1], [3, 4]),
        "rope.k": ([2, 1], [20, 21]),
        "projected.q": ([2, 2, 3], list(range(1, 13))),
        "projected.kv": ([2, 2, 3], list(range(30, 42))),
        "weight.q_projection": ([2, 3, 1], list(range(50, 56))),
        "weight.kv_projection": ([2, 3, 1], list(range(60, 66))),
        "weight.output_projection": ([1, 2], [70, 71]),
    }
    rows = []
    for semantic, (shape, words) in t.items():
        rows.append({"semantic": semantic, "dtype": "bf16", "shape": shape,
                     "file": payload(tmp_path, semantic, words)})
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "contract": {"dimensions": {"tokens": 2, "heads": 2, "qk_nope": 2,
                                      "qk_rope": 1, "v_head": 1},
                     "layout": "token-head-dense", "causal": True,
                     "softmax_scale": 0.5},
        "prompt": {"u32le_file": str(tokens)}, "tensors": rows,
    }))
    sealed = tmp_path / "sealed.json"
    subprocess.run([sys.executable, SCRIPT, "seal", "--spec", str(spec), "--output",
                    str(sealed), "--require-source"], check=True)
    out = subprocess.check_output([sys.executable, SCRIPT, "pack-materialized",
                                   "--manifest", str(sealed), "--output-dir",
                                   str(tmp_path / "packed")], text=True).strip()
    manifest = json.loads(open(out).read())
    by_name = {x["semantic"]: x for x in manifest["tensors"]}
    assert open(by_name["q"]["file"], "rb").read() == open(rows[3]["file"], "rb").read()
    assert struct.unpack("<12H", open(by_name["k"]["file"], "rb").read()) == (
        30, 31, 20, 33, 34, 20, 36, 37, 21, 39, 40, 21)
    assert struct.unpack("<4H", open(by_name["v"]["file"], "rb").read()) == (32, 35, 38, 41)
    for item in manifest["tensors"]:
        assert hashlib.sha256(open(item["file"], "rb").read()).hexdigest() == item["sha256"]


def test_validate_rejects_modified_payload(tmp_path):
    # Reuse a minimal malformed manifest to exercise the hash boundary.
    data = tmp_path / "q.bf16"
    data.write_bytes(b"\0\0")
    tokens = tmp_path / "tokens"
    tokens.write_bytes(struct.pack("<I", 1))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "plow.mla-boundary.v1",
        "contract": {"dimensions": {"tokens": 1, "heads": 1, "qk_nope": 1,
                                      "qk_rope": 1, "v_head": 1},
                     "layout": "token-head-dense", "causal": True},
        "prompt": {"u32le_file": str(tokens), "sha256": hashlib.sha256(tokens.read_bytes()).hexdigest()},
        "tensors": [{"semantic": "q", "dtype": "bf16", "shape": [1],
                     "file": str(data), "sha256": "0" * 64}],
    }))
    result = subprocess.run([sys.executable, SCRIPT, "validate", "--manifest", str(manifest)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "stale payload hash" in result.stderr


def test_attach_tensor_preserves_history_and_seals_payload(tmp_path):
    tokens = tmp_path / "tokens"
    tokens.write_bytes(struct.pack("<I", 1))
    q = tmp_path / "q.bf16"
    q.write_bytes(b"\0\0")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "plow.mla-boundary.v1",
        "contract": {"dimensions": {"tokens": 1, "heads": 1, "qk_nope": 1,
                                      "qk_rope": 1, "v_head": 1},
                     "layout": "token-head-dense", "causal": True},
        "prompt_sha256_u32le": hashlib.sha256(tokens.read_bytes()).hexdigest(),
        "prompt": {"u32le_file": str(tokens),
                   "sha256": hashlib.sha256(tokens.read_bytes()).hexdigest()},
        "tensors": [{"semantic": "q", "dtype": "bf16", "shape": [1],
                     "file": str(q), "sha256": hashlib.sha256(q.read_bytes()).hexdigest()}],
    }))
    residual = tmp_path / "residual.bf16"
    residual.write_bytes(struct.pack("<2H", 7, 8))
    output = tmp_path / "attached.json"
    subprocess.run([sys.executable, SCRIPT, "attach-tensor", "--manifest", str(manifest),
                    "--semantic", "residual.output", "--file", str(residual),
                    "--dtype", "bf16", "--shape", "1,2", "--output", str(output)], check=True)
    result = json.loads(output.read_text())
    assert result["prompt_sha256_u32le"] == hashlib.sha256(tokens.read_bytes()).hexdigest()
    item = next(x for x in result["tensors"] if x["semantic"] == "residual.output")
    assert item["sha256"] == hashlib.sha256(residual.read_bytes()).hexdigest()
