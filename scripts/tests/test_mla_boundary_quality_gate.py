import json
import struct
import subprocess
import sys


def manifest(tmp_path, name, inputs, outputs):
    tensors = []
    for semantic, values in {**inputs, **outputs}.items():
        path = tmp_path / f"{name}.{semantic}.f32"
        raw = struct.pack(f"<{len(values)}f", *values)
        path.write_bytes(raw)
        import hashlib

        tensors.append(
            {
                "semantic": semantic,
                "layer": 3,
                "rank": 0,
                "dtype": "float32",
                "source_dtype": "bf16",
                "shape": [len(values)],
                "file": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    out = tmp_path / f"{name}.json"
    out.write_text(
        json.dumps(
            {"schema": 1, "prompt_sha256_u32le": "history", "tensors": tensors}
        )
    )
    return out


def run_gate(tmp_path, ref0, ref1, absorbed, materialized):
    script = __file__.replace("tests/test_mla_boundary_quality_gate.py", "mla_boundary_quality_gate.py")
    return subprocess.run(
        [
            sys.executable,
            script,
            "--reference",
            str(ref0),
            "--reference",
            str(ref1),
            "--absorbed",
            str(absorbed),
            "--materialized",
            str(materialized),
            "--input-semantic",
            "q,k,v",
            "--output-semantic",
            "attention.output,residual.output",
            "--output",
            str(tmp_path / "result.json"),
        ],
        capture_output=True,
    )


def test_accepts_materialized_error_within_absorbed_plus_repeat_floor(tmp_path):
    inputs = {"q": [1.0], "k": [2.0], "v": [3.0]}
    ref0 = manifest(tmp_path, "r0", inputs, {"attention.output": [10.0], "residual.output": [20.0]})
    ref1 = manifest(tmp_path, "r1", inputs, {"attention.output": [10.1], "residual.output": [20.1]})
    absorbed = manifest(tmp_path, "a", inputs, {"attention.output": [10.2], "residual.output": [20.2]})
    materialized = manifest(tmp_path, "m", inputs, {"attention.output": [10.25], "residual.output": [20.25]})
    result = run_gate(tmp_path, ref0, ref1, absorbed, materialized)
    assert result.returncode == 0, result.stderr.decode()


def test_rejects_upstream_input_mismatch(tmp_path):
    inputs = {"q": [1.0], "k": [2.0], "v": [3.0]}
    ref0 = manifest(tmp_path, "r0", inputs, {"attention.output": [10.0], "residual.output": [20.0]})
    ref1 = manifest(tmp_path, "r1", inputs, {"attention.output": [10.0], "residual.output": [20.0]})
    absorbed = manifest(tmp_path, "a", inputs, {"attention.output": [10.0], "residual.output": [20.0]})
    bad = dict(inputs)
    bad["q"] = [1.5]
    materialized = manifest(tmp_path, "m", bad, {"attention.output": [10.0], "residual.output": [20.0]})
    result = run_gate(tmp_path, ref0, ref1, absorbed, materialized)
    assert result.returncode != 0
    assert b"upstream input payloads differ" in result.stderr


def test_rejects_materialized_error_beyond_control_and_floor(tmp_path):
    inputs = {"q": [1.0], "k": [2.0], "v": [3.0]}
    ref0 = manifest(tmp_path, "r0", inputs, {"attention.output": [10.0], "residual.output": [20.0]})
    ref1 = manifest(tmp_path, "r1", inputs, {"attention.output": [10.01], "residual.output": [20.01]})
    absorbed = manifest(tmp_path, "a", inputs, {"attention.output": [10.02], "residual.output": [20.02]})
    materialized = manifest(tmp_path, "m", inputs, {"attention.output": [11.0], "residual.output": [21.0]})
    result = run_gate(tmp_path, ref0, ref1, absorbed, materialized)
    assert result.returncode == 2
