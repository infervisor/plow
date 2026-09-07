#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
python3 - "$ROOT" <<'PY'
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

root = Path(sys.argv[1])
script = (root / "scripts/build_gfx942.sh").read_text()
assert "PLOW_BUILD_MIXED" not in script
rows = script.split("ROWS=(\n", 1)[1].split("\n)", 1)[0]
axes = re.search(r'"interp_mixed\|([^"]+)"', rows).group(1)
expected = set(shlex.split(axes)) | {"-DPLOW_ARCH_SUFFIX=gfx942"}

with tempfile.TemporaryDirectory(prefix="plow-mixed-build-") as tmp:
    tmp = Path(tmp)
    config = tmp / "plow_config.h"
    config.write_text("#define PLOW_PACKET_HAS_EMBED 1\n")
    for name, arch, extra in [
        ("gfx942", "gfx942", []),
        ("specialized", "gfx942", [
            f"-DPLOW_HSACO_CONFIG={config}",
            "-DPLOW_HSACO_EXTRA_DEFINES=-DPLOW_UNRELATED_PROBE=1",
            "-DPLOW_DECODE_BATCH=16",
        ]),
        ("gfx950", "gfx950", []),
    ]:
        build = tmp / name
        subprocess.run([
            "cmake", "-G", "Ninja", "-S", str(root / "runtime"), "-B", str(build),
            "-DPLOW_GFX950_HSACO=ON", f"-DPLOW_HSACO_ARCH={arch}", *extra,
        ], check=True, stdout=subprocess.DEVNULL)
        commands = [shlex.split(line.split(" = ", 1)[1])
                    for line in (build / "build.ninja").read_text().splitlines()
                    if "COMMAND = " in line and "hipcc_hsaco.sh" in line
                    and "/interp_mixed" in line]
        if arch == "gfx950":
            assert not commands, "gfx942-only mixed object leaked into gfx950"
            continue
        assert len(commands) == 2, commands
        for command in commands:
            gq = any(token.endswith("/interp_mixed_gq.elf") for token in command)
            suffix = "_gq" if gq else ""
            symbol = f"plow_interp_mixed_gfx942{suffix}"
            assert symbol in command, command
            at = command.index(symbol)
            assert command[at + 1:at + 3] == ["512", "1"], command
            defines = {token for token in command if token.startswith("-D")}
            wanted = expected | ({"-DPLOW_GLOBAL_QUEUE=1", "-DPLOW_GQ_BATCH=1"} if gq else set())
            assert defines == wanted, (name, defines ^ wanted)

print("mixed object build selftest: PASS")
PY
