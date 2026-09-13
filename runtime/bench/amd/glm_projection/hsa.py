import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--driver", type=Path, required=True)
    p.add_argument("--object", type=Path, required=True)
    p.add_argument("--direct", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    source = json.loads(a.direct.read_text())
    assert sha(a.object) == source["object_sha256"]
    results = []
    names = {"q_a": ("x.bin", "wqa.bin"), "q_absorb": ("qlat.bin", "wabs.bin"),
             "kv_latent": ("x.bin", "wkv.bin")}
    with tempfile.TemporaryDirectory(prefix="glm-lt-hsa-") as tmp:
        symbol = Path(tmp) / "symbol.txt"
        for d in source["results"]:
            if d["tuned_rows"] != (4464 if d["rows"] <= 4464 else 8192):
                continue
            symbol.write_text(d["kernel_name"])
            mi, mj, _ = re.search(r"MT(\d+)x(\d+)x(\d+)", d["kernel_name"]).groups()
            af, bf = names[d["name"]]
            expected = a.expected / (d["name"] + "-" + str(d["rows"]) + ".bin")
            command = [str(a.driver.resolve()), str(a.object.resolve()), str(symbol),
                       str(d["rows"]), str(d["n"]), str(d["k"]), mi, mj,
                       str(d["threads"]), str(d["info1"]), str(a.capture / af),
                       str(a.capture / bf), str(expected)]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f"HSA case {d['name']}/{d['rows']} failed ({completed.returncode}): {completed.stderr}")
            row = json.loads(completed.stdout)
            row.update(name=d["name"], solution_index=d["solution_index"], info1=d["info1"],
                       expected_sha256=sha(expected))
            print(json.dumps(row), flush=True)
            results.append(row)
    a.out.write_text(json.dumps(dict(object_sha256=source["object_sha256"],
        driver_sha256=sha(a.driver), direct_record_sha256=sha(a.direct),
        scope="Direct plow C HSA backend; exact full-output comparison to HIP module outputs; host enqueue+drain timings; no model serving claim",
        results=results), indent=2)+"\n")


if __name__ == "__main__":
    main()
